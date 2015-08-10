import ast
import ctypes
import random
import operator
from ctree.c.nodes import Assign, For, SymbolRef, Constant, PostInc, Lt, FunctionDecl, CFile, FunctionCall, MultiNode
from ctree.cpp.nodes import CppDefine, CppInclude
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
import math
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, \
    PyGMGOclConcreteSpecializedFunction, KernelRunManager
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, \
    CompReductionTransformer
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover, time_this, \
    flattened_to_multi_index, new_generate_control, compute_largest_local_work_size
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeGetter, \
    LookupSimplificationTransformer, AttributeRenamer, FunctionCallSimplifier, IndexTransformer, LoopUnroller, \
    IndexOpTransformer, IndexDirectTransformer, IndexOpTransformBugfixer, PyBranchSimplifier, OclFileWrapper

import numpy as np
import pycl as cl

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import dump

__author__ = 'nzhang-dev'

class RestrictCFunction(PyGMGConcreteSpecializedFunction):

    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        return (args[2].ravel(), args[3].ravel()), {}

    # def __call__(self, thing, level, target, source, restriction_type):
    #     #print(self.entry_point_name, [i.shape for i in flattened])
    #     self._c_function(target.ravel(), source.ravel())

class RestrictOclFunction(PyGMGOclConcreteSpecializedFunction):

    def set_kernel_args(self, args, kwargs):
        thing, level, target, source, restriction_type = args
        kernel = self.kernels[0]
        kernel_args = []

        for mesh in (target, source):
            if mesh.dirty:
                buffer = None if mesh.buffer is None else mesh.buffer.buffer
                buf, evt = cl.buffer_from_ndarray(self.queue, mesh, buf=buffer)
                mesh.buffer = buf
                mesh.buffer.evt = evt
                mesh.dirty = False

            elif mesh.buffer is None:
                size = mesh.size * ctypes.sizeof(ctypes.c_double)
                mesh.buffer = cl.clCreateBuffer(self.context, size)

            kernel_args.append(mesh.buffer)

        kernel.args = kernel_args

    def __call__(self, *args, **kwargs):
        self.set_kernel_args(args, kwargs)
        kernel = self.kernels[0]
        kernel_args = []
        previous_events = []
        for arg in kernel.args:
            kernel_args.append(arg.buffer)
            if arg.evt is not None:
                previous_events.append(arg.evt)

        cl.clWaitForEvents(*previous_events)
        run_evt = kernel.kernel(*kernel_args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
        run_evt.wait()

        ary, evt = cl.buffer_to_ndarray(self.queue, kernel.args[0].buffer, args[2])
        kernel.args[0].evt = evt
        kernel.args[0].dirty = False
        kernel.args[0].evt.wait()


class CRestrictSpecializer(LazySpecializedFunction):


    class RestrictSubconfig(dict):
        #hash_count = 0
        #pass
        def __hash__(self):
            hash_thing = (
                self['level'].space,
                self['level'].ghost_zone,
                self['self'].neighbor_offsets,
                self['source'].shape,
                self['target'].shape,
                self['restriction_type'],
            )
            return hash(hash_thing)
            #print(hash_thing)
        #     #self.hash_count += 1
        #     return id(self)

    def args_to_subconfig(self, args):
        return self.RestrictSubconfig({
            'self': args[0],
            'level': args[1],
            'target': args[2],
            'source': args[3],
            'restriction_type': args[4]
        })


    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        ndim = subconfig['self'].dimensions
        layers = [
            ParamStripper(('self', 'level', 'restriction_type')),
            AttributeRenamer({'restriction_type': ast.Num(n=subconfig['restriction_type'])}),
            AttributeGetter({'self': subconfig['self']}),
            PyBranchSimplifier(),
            SemanticFinder(subconfig, locals=subconfig),
            IndexTransformer(('target_point', 'source_point')),
            AttributeGetter(subconfig),
            LookupSimplificationTransformer(),
            FunctionCallSimplifier(),
            LoopUnroller(),
            GeneratorTransformer(subconfig),
            CompReductionTransformer(),
            IndexOpTransformer(ndim=ndim, encode_func_names={'target_point': 'target_encode', 'source_point': 'source_encode'}),
            IndexDirectTransformer(ndim=ndim, encode_func_names={'source_point': 'source_encode', 'target_point': 'target_encode'}),
            IndexOpTransformBugfixer(func_names=('target_encode', 'source_encode')),
            CRangeTransformer() if subconfig['level'].configuration.backend != 'ocl' else self.RangeTransformer(),
            PyBasicConversions(),
        ]
        tree = apply_all_layers(layers, tree)
        #print(dump(tree))
        function = tree.find(FunctionDecl)
        function.defn = [
            SymbolRef('source_point_{}'.format(i), sym_type=ctypes.c_uint64())
            for i in range(ndim)
        ] + function.defn
        function.name = 'totally_not_restrict'
        for param in function.params:
            param.type = ctypes.POINTER(ctypes.c_double)()
        #print(dump(tree))
        ordering = Ordering([MultiplyEncode()], prefix='source_')
        bits_per_dim = min([math.log(i, 2) for i in subconfig['source'].shape]) + 1
        encode_func_source = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        ordering = Ordering([MultiplyEncode()], prefix='target_')
        bits_per_dim = min([math.log(i, 2) for i in subconfig['target'].shape]) + 1
        encode_func_target = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        cfile = CFile(body=[tree, encode_func_source, encode_func_target])
        cfile = include_mover(cfile)
        #print(subconfig['self'].neighbor_offsets[subconfig['restriction_type']])
        # if subconfig['restriction_type'] == 0:
        #     print(cfile)
        #print(subconfig['target'].shape, subconfig['source'].shape)
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = RestrictCFunction()
        subconfig, tuner_config = program_config
        np_arrays = subconfig['target'], subconfig['source']
        param_types = [
            np.ctypeslib.ndpointer(arr.dtype, 1, arr.size) for arr in np_arrays
        ]
        name = self.tree.body[0].name
        return fn.finalize('totally_not_restrict', Project(transform_result),
                           ctypes.CFUNCTYPE(None, *param_types))


class OclRestrictSpecializer(CRestrictSpecializer):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            body=[
                Assign(SymbolRef("global_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
            ]
            ranges = node.iterator.ranges
            offsets = tuple(r[0] for r in ranges)
            shape = tuple(r[1] - r[0] for r in ranges)
            indices = flattened_to_multi_index(SymbolRef("global_id"), shape, offsets=offsets)
            for d in range(len(shape)):
                body.append(Assign(SymbolRef("target_point_%d"%d, ctypes.c_ulong()), indices[d]))
            body.extend(node.body)
            return MultiNode(body=body)

    def transform(self, tree, program_config):
        subconfig, tuner = program_config
        level = subconfig['level']
        interior_space = level.interior_space
        file = super(OclRestrictSpecializer, self).transform(tree, program_config)[0]
        while (isinstance(file.body[0], CppInclude)):
            file.body.pop(0)
        kernel = file.find(FunctionDecl)
        kernel.set_kernel()
        for param in kernel.params:
            param.set_global()
        ocl_file = OclFileWrapper("%s_kernel" % kernel.name).visit(file)
        global_size = reduce(operator.mul, interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        control = new_generate_control("%s_control" % kernel.name, global_size, local_size, kernel.params, [kernel])
        kernel.name = "%s_kernel" % kernel.name
        return [control, ocl_file]

    def finalize(self, transform_result, program_config):
        subconfig, tuner = program_config
        level = subconfig['level']
        project = Project(transform_result)
        kernel = transform_result[1]
        control = transform_result[0]
        name = kernel.name
        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[name]
        kernel.argtypes = (cl.cl_mem, cl.cl_mem)
        global_size = reduce(operator.mul, level.interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], )
        kernel = KernelRunManager(kernel, global_size, local_size)

        typesig = [ctypes.c_int, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem, cl.cl_mem]
        fn = RestrictOclFunction()
        fn.finalize(control.name, project, ctypes.CFUNCTYPE(*typesig),
                    level.context, level.queue, [kernel])
        return fn

