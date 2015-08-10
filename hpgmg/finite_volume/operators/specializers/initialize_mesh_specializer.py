import ast
import ctypes
from ctree.c.nodes import MultiNode, For, Assign, SymbolRef, Constant, Lt, PostInc, FunctionCall, FunctionDecl, CFile
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.ocl.nodes import OclFile
from ctree.transformations import PyBasicConversions
import math
from ctree.transforms.declaration_filler import DeclarationFiller
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, KernelRunManager, \
    PyGMGOclConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, to_macro_function, sympy_to_c, \
    include_mover, flattened_to_multi_index, new_generate_control, compute_largest_local_work_size
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder

from ctree.frontend import dump, get_ast
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeGetter, \
    IndexOpTransformer, AttributeRenamer, CallReplacer, IndexDirectTransformer, IndexTransformer, OclFileWrapper

import numpy as np
import operator
import pycl as cl

__author__ = 'nzhang-dev'


class InitializeCFunction(PyGMGConcreteSpecializedFunction):
    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        return [args[2].ravel()], {}

    # def __call__(self, thing, level, mesh, exp, coord_transform):
    #     return self._c_function(mesh.ravel())


class InitializeOclFunction(PyGMGOclConcreteSpecializedFunction):

    def set_kernel_args(self, args, kwargs):
        thing, level, mesh, exp, coord_transform = args
        kernel = self.kernels[0]

        if mesh.dirty:
            buffer = None if mesh.buffer is None else mesh.buffer.buffer
            buf, evt = cl.buffer_from_ndarray(self.queue, mesh, buf=buffer)
            mesh.buffer = buf
            mesh.buffer.evt = evt
            mesh.dirty = False

        elif mesh.buffer is None:
            size = mesh.size * ctypes.sizeof(ctypes.c_double)
            mesh.buffer = cl.clCreateBuffer(self.context, size)


        kernel.args = [mesh.buffer]

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


class CInitializeMesh(LazySpecializedFunction):

    class InitializeMeshSubconfig(dict):
        def __hash__(self):
            to_hash = (
                self['level'].space,
                self['level'].ghost_zone,
                self['mesh'].shape,
                str(self['exp']),
                self['coord_transform'].__name__
            )
            return hash(to_hash)

    def args_to_subconfig(self, args):
        return self.InitializeMeshSubconfig({
            'self': args[0],
            'level': args[1],
            'mesh': args[2],
            'exp': args[3],
            'coord_transform': args[4]
        })

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        func_node = tree.body[0]
        func_node.body.pop(0)
        ndim = subconfig['self'].dimensions

        coord_tree = get_ast(subconfig['coord_transform']).body[0].body[-1].value

        coord_layers = [
            SemanticFinder(),
            AttributeGetter({'self': subconfig['level']}),
            IndexOpTransformer(ndim=ndim, encode_func_names={'coord': ''}),
            PyBasicConversions(),
        ]

        coord_tree = apply_all_layers(coord_layers, coord_tree)
        coord_transform = coord_tree.args
        expr = sympy_to_c(subconfig['exp'], 'coord_')

        expr_layers = [
            AttributeRenamer({
                'coord_{}'.format(i) : coord_transform[i] for i in range(ndim)
            })
        ]

        expr = apply_all_layers(expr_layers, expr)

        layers = [
            ParamStripper(('self', 'level', 'exp', 'coord_transform')),
            SemanticFinder(subconfig),
            CallReplacer({
                'func': expr
            }),
            CRangeTransformer() if subconfig['self'].configuration.backend != "ocl" else self.RangeTransformer(),
            IndexTransformer(('coord',)),
            IndexDirectTransformer(ndim=ndim, encode_func_names={'coord': 'encode'}),
            PyBasicConversions(),

        ]

        tree = apply_all_layers(layers, tree)
        func_node = tree.find(FunctionDecl)
        func_node.params[0].type = ctypes.POINTER(ctypes.c_double)()

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)

        cfile = CFile(body=[
            tree,
            CppInclude("math.h"),
            encode_func
        ])

        cfile = include_mover(cfile)
        print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        fn = InitializeCFunction()
        return fn.finalize(
            self.tree.body[0].name,
            Project(transform_result),
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(
                subconfig['mesh'].dtype,
                1,
                subconfig['mesh'].size
            ))
        )


class OclInitializeMesh(CInitializeMesh):
    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            body=[
                Assign(SymbolRef("global_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
            ]
            ranges = node.iterator.ranges
            shape = tuple(r[1] - r[0] for r in ranges)
            indices = flattened_to_multi_index(SymbolRef("global_id"), shape)
            for d in range(len(shape)):
                body.append(Assign(SymbolRef("coord_%d"%d), indices[d]))
            body.extend(node.body)
            return MultiNode(body=body)

    def transform(self, tree, program_config):
        subconfig, tuner = program_config
        level = subconfig['level']
        global_size = reduce(operator.mul, level.space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)

        file = super(OclInitializeMesh, self).transform(tree, program_config)[0]
        kernel = file.find(FunctionDecl)

        while isinstance(file.body[0], CppInclude):
            file.body.pop(0)
        ocl_file = OclFileWrapper("%s_kernel" % kernel.name).visit(file)
        ocl_file = DeclarationFiller().visit(ocl_file)
        kernel.set_kernel()
        for param in kernel.params:
            param.set_global()
        control = new_generate_control("%s_control" % kernel.name, global_size, local_size, kernel.params, [ocl_file])
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
        kernel.argtypes = (cl.cl_mem,)
        global_size = reduce(operator.mul, level.space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        kernel = KernelRunManager(kernel, global_size, local_size)

        typesig = [ctypes.c_int, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem]
        fn = InitializeOclFunction()
        fn.finalize(control.name, project, ctypes.CFUNCTYPE(*typesig),
                    level.context, level.queue, [kernel])
        return fn
