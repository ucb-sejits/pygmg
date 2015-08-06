import ast
from collections import OrderedDict
import ctypes
from ctree.c.macros import NULL
from ctree.c.nodes import SymbolRef, Constant, PostInc, Lt, For, Assign, FunctionDecl, CFile, Return, FunctionCall, \
    MultiNode, ArrayRef, Array, Ref
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.smooth_specializer import apply_all_layers
from hpgmg.finite_volume.operators.specializers.util import include_mover, flattened_to_multi_index
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops

from ctree.frontend import dump
import numpy as np
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexDirectTransformer, \
    IndexTransformer, OclFileWrapper
# from hpgmg.finite_volume.simple_level import SimpleLevel
import operator
import pycl as cl

__author__ = 'nzhang-dev'

class MeshOpCFunction(PyGMGConcreteSpecializedFunction):
    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        flattened = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                flattened.append(arg.ravel())
            elif isinstance(arg, (int, float)):
                flattened.append(arg)
        return flattened, {}


# class MeshOpOclFunction(PyGMGConcreteSpecializedFunction):
#
#     def __init__(self, kernels):
#         self.kernels = kernels
#
#     @staticmethod
#     def pyargs_to_cargs(*args, **kwargs):
#         flattened = []
#         for arg in args:
#             if isinstance(arg, hpgmg.finite_volume.simple_level.SimpleLevel):
#                 flattened.append(arg.buffers[0])
#             elif isinstance(arg, (int, float)):
#                 flattened.append(arg)
#         # for kernel in self.kernels:
#         #     flattened.append(kernel)
#         return flattened, {}
#
#     def __call__(self, *args, **kwargs):
#         result = self.pyargs_to_cargs(args, kwargs)
#         cargs, ckwargs = result
#         for kernel in self.kernel:
#             cargs.append(kernel)
#         return self._c_function(*cargs, **ckwargs)


class MeshOpSpecializer(LazySpecializedFunction):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['{}_{}'.format(node.target, i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            bottom.body = node.body
            self.generic_visit(bottom)
            return top

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        ndim = subconfig['self'].solver.dimensions
        layers = [
            ParamStripper(('self')),
            SemanticFinder(subconfig),
            CRangeTransformer() if subconfig['self'].configuration.backend != 'ocl' else self.RangeTransformer(),
            IndexTransformer(('index')),
            IndexDirectTransformer(ndim, {'index': 'encode'}),
            PyBasicConversions()
        ]

        tree = apply_all_layers(layers, tree)

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['self'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        tree.find(CFile).body.append(encode_func)
        tree = include_mover(tree)
        return [tree]


class CFillMeshSpecializer(MeshOpSpecializer):

    class MeshSubconfig(dict):
        def __hash__(self):
            return hash(tuple(self[key].shape for key in ('mesh',)))

    def args_to_subconfig(self, args):
        return self.MeshSubconfig({
            key: arg for key, arg in zip(('self', 'mesh', 'value'), args)
        })

    def transform(self, f, program_config):
        f = super(CFillMeshSpecializer, self).transform(f, program_config)[0]
        #print(f)
        func_decl = f.find(FunctionDecl)
        param_types = [ctypes.POINTER(ctypes.c_double)(), ctypes.c_double()]
        for param, t in zip(func_decl.params, param_types):
            param.type = t

        f = include_mover(f)

        return [f]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        tree = transform_result[0]
        retval = None
        entry_point = self.tree.body[0].name
        param_types = []
        for param in (('mesh', 'value')):
            if isinstance(subconfig[param], np.ndarray):
                arr = subconfig[param]
                param_types.append(np.ctypeslib.ndpointer(
                    arr.dtype, 1, arr.size
                ))
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)

        fn = MeshOpCFunction()
        return fn.finalize(
            entry_point, Project(transform_result), ctypes.CFUNCTYPE(retval, *param_types)
        )


# class OclFillMeshSpecializer(MeshOpSpecializer):
#
#     class RangeTransformer(ast.NodeTransformer):
#         def visit_RangeNode(self, node):
#             body=[
#                 Assign(SymbolRef("global_id", ctypes.c_int()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
#             ]
#             ranges = node.iterator.ranges
#             offsets = tuple(r[0] for r in ranges)
#             shape = tuple(r[1] - r[0] for r in ranges)
#             indices = flattened_to_multi_index(SymbolRef("global_id"), shape, offsets=offsets)
#             for d in range(len(shape)):
#                 body.append(Assign(SymbolRef("index_%d"%d, ctypes.c_int()), indices[d]))
#             body.extend(node.body)
#             return MultiNode(body=body)
#
#     class MeshSubconfig(dict):
#         def __hash__(self):
#             return hash(tuple(self[key].shape for key in ('mesh',)))
#
#     def args_to_subconfig(self, args):
#         return self.MeshSubconfig({
#             key: arg for key, arg in zip(('self', 'mesh', 'value'), args)
#         })
#
#     def transform(self, f, program_config):
#         subconfig, tuner = program_config
#         f = super(OclFillMeshSpecializer, self).transform(f, program_config)[0]
#         func_decl = f.find(FunctionDecl)
#         param_types = [ctypes.POINTER(ctypes.c_double)(), ctypes.c_double()]
#         for param, t in zip(func_decl.params, param_types):
#             param.type = t
#             if isinstance(t, ctypes.POINTER(ctypes.c_double)):
#                 param.set_global()
#         func_decl.set_kernel()
#         f = include_mover(f)
#         # remove includes
#         while isinstance(f.body[0], CppInclude):
#             f.body.pop(0)
#         kernel = OclFileWrapper(name="fill_mesh_kernel").visit(f)
#         global_shape = tuple(dim + 2 * ghost for dim, ghost in zip(subconfig['self'].interior_space, subconfig['self'].ghost_zone))
#         control = generate_control(global_shape)
#         print(control)
#         print(kernel)
#         return [control, kernel]
#
#     def finalize(self, transform_result, program_config):
#         subconfig, tuner_config = program_config
#         tree = transform_result[0]
#         retval = None
#         entry_point = self.tree.body[0].name + "_control"
#         param_types = [cl.cl_command_queue]
#         for param in (('mesh', 'value')):
#             if isinstance(subconfig[param], np.ndarray):
#                 # arr = subconfig[param]
#                 # param_types.append(np.ctypeslib.ndpointer(
#                 #     arr.dtype, 1, arr.size
#                 # ))
#                 param_types.append(cl.cl_mem)
#             elif isinstance(subconfig[param], (int, float)):
#                 param_types.append(ctypes.c_double)
#         param_types.append(cl.cl_kernel)
#         level = subconfig['self']
#         kernel = transform_result[1]
#         kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()["fill_mesh_kernel"]
#         fn = MeshOpOclFunction([kernel])
#         return fn.finalize(
#             entry_point, Project(transform_result), ctypes.CFUNCTYPE(retval, *param_types)
#         )


class CGeneralizedSimpleMeshOpSpecializer(MeshOpSpecializer):

    class GeneralizedSubconfig(OrderedDict):
        def __hash__(self):
            to_hash = []
            for key, value in self.items():
                if isinstance(value, np.ndarray):
                    to_hash.append((key, value.shape))
            return hash(tuple(to_hash))

    def args_to_subconfig(self, args):
        argument_names = [arg.id for arg in self.tree.body[0].args.args]
        retval = self.GeneralizedSubconfig()
        for key, val in ((argument_name, arg) for argument_name, arg in zip(argument_names, args)):
            retval[key] = val
        return retval

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        f = super(CGeneralizedSimpleMeshOpSpecializer, self).transform(tree, program_config)[0]
        decl = f.find(FunctionDecl)
        params = []
        for param in decl.params:
            if not isinstance(subconfig[param.name], (int, float, np.ndarray)):
                continue
            if isinstance(subconfig[param.name], (int, float)):
                param.type = ctypes.c_double()
            else:
                param.type = ctypes.POINTER(ctypes.c_double)()
                param.set_restrict()
            params.append(param)
        decl.params = params
        if decl.find(Return):
            decl.return_type = ctypes.c_double()

        for call in decl.find_all(FunctionCall):
            if call.func.name == 'abs':
                call.func.name = 'fabs'
                f.body.append(CppInclude("math.h"))
        #print(f)
        f = include_mover(f)
        return [f]


    def finalize(self, transform_result, program_config):
        fn = MeshOpCFunction()
        subconfig, tuner_config = program_config
        param_types = []
        for key, value in subconfig.items():
            if isinstance(value, (int, float)):
                param_types.append(ctypes.c_double)
            if isinstance(value, np.ndarray):
                param_types.append(np.ctypeslib.ndpointer(value.dtype, 1, value.size))

        name = self.tree.body[0].name
        if any(isinstance(i, ast.Return) for i in ast.walk(self.tree)):
            return_type = ctypes.c_double
        else:
            return_type = None

        return fn.finalize(
            name, Project(transform_result), ctypes.CFUNCTYPE(return_type, *param_types)
        )


def generate_control(global_shape):
    ndim = len(global_shape)
    global_size = reduce(operator.mul, global_shape, 1)
    local_size = min(1024, global_size)

    defn = [
        Assign((ArrayRef(SymbolRef("global", ctypes.c_ulong()), Constant(1))), Array(body=[Constant(global_size)])),
        Assign((ArrayRef(SymbolRef("local", ctypes.c_ulong()), Constant(1))), Array(body=[Constant(local_size)])),
        FunctionCall(SymbolRef("clSetKernelArg"),
                               [
                                   SymbolRef("fill_mesh_kernel"),
                                   Constant(0),
                                   FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
                                   Ref(SymbolRef("mesh"))
                               ]),
        FunctionCall(SymbolRef("clSetKernelArg"),
                               [
                                   SymbolRef("fill_mesh_kernel"),
                                   Constant(1),
                                   Constant(ctypes.sizeof(ctypes.c_double())),
                                   SymbolRef("value")
                               ]),
        FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
            SymbolRef("queue"),
            SymbolRef("fill_mesh_kernel"),
            Constant(1),
            NULL(),
            SymbolRef("global"),
            SymbolRef("local"),
            Constant(0),
            NULL(),
            NULL()
        ]),
        Return(Constant(0))
    ]

    control_params = []

    control_params.append(SymbolRef("queue", cl.cl_command_queue()))
    control_params.append(SymbolRef("mesh", cl.cl_mem()))
    control_params.append(SymbolRef("value", ctypes.c_double()))
    control_params.append(SymbolRef("fill_mesh_kernel", cl.cl_kernel()))

    control_func = FunctionDecl(return_type=ctypes.c_int(), name="fill_mesh_control", params=control_params, defn=defn)

    body = [control_func]

    return CFile(name="fill_mesh_control", body=body, config_target='opencl')