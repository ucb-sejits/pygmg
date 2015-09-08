import ast
import ctypes
from ctree.c.nodes import Assign, For, SymbolRef, Constant, Lt, PostInc, CFile, FunctionDecl, FunctionCall, MultiNode
from ctree.cpp.nodes import CppInclude
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from ctree.transforms.declaration_filler import DeclarationFiller
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, \
    PyGMGOclConcreteSpecializedFunction, KernelRunManager
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover, new_generate_control, \
    compute_largest_local_work_size, flattened_to_multi_index
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.semantic_transformers.oclsemantics import OclRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import AttributeRenamer, AttributeGetter, \
    ParamStripper, ArrayRefIndexTransformer, IndexOpTransformer, IndexTransformer, IndexDirectTransformer, \
    IndexOpTransformBugfixer, OclFileWrapper

import numpy as np
import pycl as cl
import operator

from ctree.frontend import dump

__author__ = 'nzhang-dev'

class InterpolateCFunction(PyGMGConcreteSpecializedFunction):
    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        target_mesh, source_mesh = args[-2:]
        return (target_mesh.ravel(), source_mesh.ravel()), {}

    # def __call__(self, thing, target_level, target_mesh, source_mesh):
    #     args = [
    #         target_mesh,
    #         source_mesh
    #     ]
    #     flattened = [i.ravel() for i in args]
    #     return self._c_function(*flattened)


class InterpolateOclFunction(PyGMGOclConcreteSpecializedFunction):

    def set_dirty_buffers(self, args):
        args[2].buffer.dirty = True


class CInterpolateSpecializer(LazySpecializedFunction):

    def args_to_subconfig(self, args):
        return {
            'self': args[0],
            'target_level': args[1],
            'target_mesh': args[2],
            'source_mesh': args[3]
        }

    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['self'].dimensions
        #shape = subconfig['self'].operator.
        layers = [
            ParamStripper(('self', 'target_level')),
            # AttributeRenamer({
            #     'self.operator.apply_op': ast.Name('apply_op', ast.Load())
            # }),
            SemanticFinder(subconfig),
            AttributeGetter(subconfig),
            CRangeTransformer(),
            IndexTransformer(('target_index', 'source_index')),
            IndexOpTransformer(ndim, {'target_index': 'target_encode', 'source_index': 'source_encode'}),
            ArrayRefIndexTransformer(
                encode_map={
                    'target_index': 'target_encode',
                    'source_index': 'source_encode'
                },
                ndim=ndim
            ),
            IndexDirectTransformer(ndim, encode_func_names={'target_index': 'target_encode', 'source_index': 'source_encode'}),
            IndexOpTransformBugfixer(func_names=('target_encode', 'source_encode')),
            PyBasicConversions(),
        ]
        func = apply_all_layers(layers, func)
        for param in func.params:
            param.type = ctypes.POINTER(ctypes.c_double)()

        # func.defn = [
        #     SymbolRef('source_index', sym_type=ctypes.c_uint64()),
        #     SymbolRef('target_index', sym_type=ctypes.c_uint64())
        # ] + func.defn

        ordering = Ordering([MultiplyEncode()], prefix="source_")
        source_bits_per_dim = min([math.log(i, 2) for i in subconfig['source_mesh'].space]) + 1
        target_bits_per_dim = min([math.log(i, 2) for i in subconfig['target_mesh'].space]) + 1
        source_encode = ordering.generate(ndim, source_bits_per_dim, ctypes.c_uint64)
        ordering.prefix = 'target_'
        target_encode = ordering.generate(ndim, target_bits_per_dim, ctypes.c_uint64)

        # print(source_encode)
        # print(target_encode)
        # print(func)

        cfile = CFile(body=[
            source_encode,
            target_encode,
            func
        ])
        cfile = include_mover(cfile)
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = InterpolateCFunction()
        subconfig = program_config[0]
        name = self.tree.body[0].name
        ctype = [np.ctypeslib.ndpointer(
                subconfig[key].dtype,
                1,
                subconfig[key].size
            ) for key in ('target_mesh', 'source_mesh')]
        return fn.finalize(
            name,
            Project(transform_result),
            ctypes.CFUNCTYPE(None, *ctype)
        )


class OclInterpolateSpecializer(LazySpecializedFunction):

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
                body.append(Assign(SymbolRef("target_index_%d"%d, ctypes.c_ulong()), indices[d]))
            body.extend(node.body)
            return MultiNode(body=body)

    def args_to_subconfig(self, args):
        return {
            'self': args[0],
            'target_level': args[1],
            'target_mesh': args[2],
            'source_mesh': args[3]
        }

    def transform(self, tree, program_config):
        subconfig, tuner = program_config
        target_level = subconfig['target_level']
        ndim = subconfig['self'].dimensions

        func = tree.body[0]
        layers = [
            ParamStripper(('self', 'target_level')),
            SemanticFinder(subconfig),
            AttributeGetter(subconfig),
            # self.RangeTransformer(),
            OclRangeTransformer(),
            IndexTransformer(('target_index', 'source_index')),
            IndexOpTransformer(ndim, {'target_index': 'target_encode', 'source_index': 'source_encode'}),
            ArrayRefIndexTransformer(
                encode_map={
                    'target_index': 'target_encode',
                    'source_index': 'source_encode'
                },
                ndim=ndim
            ),
            IndexDirectTransformer(ndim, encode_func_names={'target_index': 'target_encode', 'source_index': 'source_encode'}),
            IndexOpTransformBugfixer(func_names=('target_encode', 'source_encode')),
            PyBasicConversions(),
        ]
        func = apply_all_layers(layers, func)
        for param in func.params:
            param.type = ctypes.POINTER(ctypes.c_double)()

        ordering = Ordering([MultiplyEncode()], prefix="source_")
        source_bits_per_dim = min([math.log(i, 2) for i in subconfig['source_mesh'].space]) + 1
        target_bits_per_dim = min([math.log(i, 2) for i in subconfig['target_mesh'].space]) + 1
        source_encode = ordering.generate(ndim, source_bits_per_dim, ctypes.c_uint64)
        ordering.prefix = 'target_'
        target_encode = ordering.generate(ndim, target_bits_per_dim, ctypes.c_uint64)

        cfile = CFile(body=[
            source_encode,
            target_encode,
            func
        ])
        cfile = include_mover(cfile)

        while isinstance(cfile.body[0], CppInclude):
            cfile.body.pop(0)
        kernel = cfile.find(FunctionDecl)
        kernel.set_kernel()
        for param in kernel.params:
            param.set_global()
        ocl_file = OclFileWrapper("%s_kernel" % kernel.name).visit(cfile)
        ocl_file = DeclarationFiller().visit(ocl_file)
        global_size = reduce(operator.mul, target_level.interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        control = new_generate_control("%s_control" % kernel.name, global_size, local_size, kernel.params, [kernel])
        kernel.name = "%s_kernel" % kernel.name
        # print(control)
        # raise TypeError
        return [control, ocl_file]
        # return [ocl_file]

    def finalize(self, transform_result, program_config):
        subconfig, tuner = program_config
        target_level = subconfig['target_level']
        global_size = reduce(operator.mul, target_level.interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)

        project = Project(transform_result)
        control = transform_result[0]
        kernel = transform_result[1]

        name = kernel.name
        kernel = cl.clCreateProgramWithSource(target_level.context, kernel.codegen()).build()[name]
        kernel.argtypes = (cl.cl_mem, cl.cl_mem)
        kernel = KernelRunManager(kernel, global_size, local_size)

        typesig = [ctypes.c_int, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem, cl.cl_mem]
        typesig.append(np.ctypeslib.ndpointer(np.float32, 1, (1,)))
        fn = InterpolateOclFunction()
        fn = fn.finalize(control.name, project, ctypes.CFUNCTYPE(*typesig),
                         target_level, [kernel])
        # fn.finalize(project, target_level, [kernel])
        return fn
