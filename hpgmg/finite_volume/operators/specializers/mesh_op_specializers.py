import ast
from collections import OrderedDict
import ctypes
from ctree.c.macros import NULL
from ctree.c.nodes import SymbolRef, Constant, PostInc, Lt, For, Assign, FunctionDecl, CFile, Return, FunctionCall, \
    MultiNode, ArrayRef, Array, Ref, Mul, Add, If, Gt
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
import math
from ctree.transforms.declaration_filler import DeclarationFiller
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.smooth_specializer import apply_all_layers
from hpgmg.finite_volume.operators.specializers.util import include_mover, flattened_to_multi_index, \
    new_generate_control
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


class MeshOpOclFunction(ConcreteSpecializedFunction):

    def __init__(self, kernels, other_args=None):
        self.kernels = kernels
        self.other_args = other_args if other_args else []

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self.entry_point_name = entry_point_name
        self.entry_point_typesig = entry_point_typesig
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        return self

    def __call__(self, *args, **kwargs):
        arguments = []
        queue = None
        args = args + tuple(self.other_args)
        for arg in args:
            if hasattr(arg, "queue"):
                queue = arg.queue
                arguments.append(queue)
                arguments.extend(kernel for kernel in self.kernels)
            elif isinstance(arg, Mesh) or isinstance(arg, np.ndarray):
                if hasattr(arg, "buffer"):
                    if arg.buffer is None:
                        arg.buffer, evt = cl.buffer_from_ndarray(queue, arg)
                    else:
                        arg.buffer, evt = cl.buffer_from_ndarray(queue, arg, buf=arg.buffer)
                    arguments.append(arg.buffer)
                else:
                    buf, evt = cl.buffer_from_ndarray(queue, arg)
                    arguments.append(buf)
            elif isinstance(arg, (int, float)):
                arguments.append(arg)

        value = self._c_function(*arguments)
        for arg in args:
            if isinstance(arg, Mesh):
                arg, evt = cl.buffer_to_ndarray(queue, arg.buffer, out=arg)

        return value


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


class OclFillMeshSpecializer(MeshOpSpecializer):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            body=[
                Assign(SymbolRef("global_id", ctypes.c_int()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
            ]
            ranges = node.iterator.ranges
            offsets = tuple(r[0] for r in ranges)
            shape = tuple(r[1] - r[0] for r in ranges)
            indices = flattened_to_multi_index(SymbolRef("global_id"), shape, offsets=offsets)
            for d in range(len(shape)):
                body.append(Assign(SymbolRef("index_%d"%d, ctypes.c_int()), indices[d]))
            body.extend(node.body)
            return MultiNode(body=body)

    class MeshSubconfig(dict):
        def __hash__(self):
            return hash(tuple(self[key].shape for key in ('mesh',)))

    def args_to_subconfig(self, args):
        return self.MeshSubconfig({
            key: arg for key, arg in zip(('self', 'mesh', 'value'), args)
        })

    def transform(self, f, program_config):
        subconfig, tuner = program_config
        f = super(OclFillMeshSpecializer, self).transform(f, program_config)[0]
        func_decl = f.find(FunctionDecl)
        param_types = [ctypes.POINTER(ctypes.c_double)(), ctypes.c_double()]
        for param, t in zip(func_decl.params, param_types):
            param.type = t
            if isinstance(t, ctypes.POINTER(ctypes.c_double)):
                param.set_global()
        func_decl.set_kernel()
        func_name = func_decl.name
        func_decl.name = "%s_kernel" % func_decl.name
        f = include_mover(f)
        # remove includes
        while isinstance(f.body[0], CppInclude):
            f.body.pop(0)
        kernel = OclFileWrapper(name=func_decl.name).visit(f)
        global_shape = tuple(dim + 2 * ghost for dim, ghost in zip(subconfig['self'].interior_space, subconfig['self'].ghost_zone))
        # control = generate_control(global_shape)
        global_size = reduce(operator.mul, global_shape, 1)
        local_size = min(1024, global_size)
        while global_size % local_size != 0:
            local_size -= 1
        control = new_generate_control("%s_control" % func_name, global_size, local_size, func_decl.params, [func_decl])
        # print(control)
        # print(kernel)
        return [control, kernel]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        project = Project(transform_result)
        control = transform_result[0]
        kernel = transform_result[1]
        retval = ctypes.c_int
        kernel_name = self.tree.body[0].name  # refers to original FunctionDef
        entry_point = kernel_name + "_control"
        param_types = [cl.cl_command_queue, cl.cl_kernel]
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        level = subconfig['self']
        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel_name + "_kernel"]
        fn = MeshOpOclFunction([kernel])
        return fn.finalize(
            entry_point, project, ctypes.CFUNCTYPE(retval, *param_types)
        )


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


class OclGeneralizedSimpleMeshOpSpecializer(MeshOpSpecializer):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            body=[
                Assign(SymbolRef("global_id", ctypes.c_int()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
            ]
            ranges = node.iterator.ranges
            offsets = tuple(r[0] for r in ranges)
            shape = tuple(r[1] - r[0] for r in ranges)
            indices = flattened_to_multi_index(SymbolRef("global_id"), shape, offsets=offsets)
            for d in range(len(shape)):
                body.append(Assign(SymbolRef("index_%d"%d, ctypes.c_int()), indices[d]))
            body.extend(node.body)
            return MultiNode(body=body)

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

    def transform(self, f, program_config):
        subconfig, tuner = program_config
        f = super(OclGeneralizedSimpleMeshOpSpecializer, self).transform(f, program_config)[0]
        func_decl = f.find(FunctionDecl)
        params = []
        for param in func_decl.params:
            if not isinstance(subconfig[param.name], (int, float, np.ndarray)):
                continue
            if isinstance(subconfig[param.name], (int, float)):
                param.type = ctypes.c_double()
            else:
                param.type = ctypes.POINTER(ctypes.c_double)()
                # param.set_restrict()
                param.set_global()
            params.append(param)
        func_decl.params = params

        func_decl.set_kernel()
        func_name = func_decl.name
        func_decl.name = "%s_kernel" % func_decl.name



        f = include_mover(f)
        # remove includes
        while isinstance(f.body[0], CppInclude):
            f.body.pop(0)
        kernel = OclFileWrapper(name=func_decl.name).visit(f)
        global_shape = tuple(dim + 2 * ghost for dim, ghost in zip(subconfig['self'].interior_space, subconfig['self'].ghost_zone))
        global_size = reduce(operator.mul, global_shape, 1)
        local_size = min(1024, global_size)
        while global_size % local_size != 0:
            local_size -= 1
        control = new_generate_control("%s_control" % func_name, global_size, local_size, func_decl.params, [func_decl])
        # print(control)
        # print(kernel)
        return [control, kernel]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        project = Project(transform_result)
        control = transform_result[0]
        kernel = transform_result[1]
        retval = ctypes.c_int
        kernel_name = self.tree.body[0].name  # refers to original FunctionDef
        entry_point = kernel_name + "_control"
        param_types = [cl.cl_command_queue, cl.cl_kernel]
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        level = subconfig['self']
        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel_name + "_kernel"]
        fn = MeshOpOclFunction([kernel])
        return fn.finalize(
            entry_point, project, ctypes.CFUNCTYPE(retval, *param_types)
        )


# class OclNormMeshSpecializer(OclGeneralizedSimpleMeshOpSpecializer):
#
#     class RangeTransformer(ast.NodeTransformer):
#         def visit_RangeNode(self, node):
#
#             ranges = node.iterator.ranges
#             offsets = tuple(r[0] for r in ranges)
#             shape = tuple(r[1] - r[0] for r in ranges)
#             ndim = len(shape)
#             global_size = reduce(operator.mul, shape, 1)  # this is an interior space
#             if global_size > cl.clGetDeviceIDs()[-1].max_work_group_size:
#                 stride = global_size / cl.clGetDeviceIDs()[-1].max_work_group_size
#             else:
#                 stride = 1
#
#             body = []
#             global_id = FunctionCall(SymbolRef("get_global_id"), [Constant(0)])
#             body.append(Assign(SymbolRef("global_offset", ctypes.c_int()), Mul(global_id, Constant(stride))))
#             body.append(SymbolRef("____temp__max_norm", ctypes.c_double()))
#
#             body.extend(SymbolRef("index_%d"%d, ctypes.c_int()) for d in range(ndim))
#
#             indices = flattened_to_multi_index(Add(SymbolRef("global_offset"), SymbolRef("local_offset")),
#                                                shape, offsets=offsets)
#
#             loop_body = []
#             loop_body.extend(Assign(SymbolRef("index_%d"%d), indices[d]) for d in range(ndim))
#             loop_body.extend(node.body)
#
#             for_loop = For(init=Assign(SymbolRef("local_offset", ctypes.c_int()), Constant(0)),
#                            test=Lt(SymbolRef("local_offset"), Constant(stride)),
#                            incr=PostInc(SymbolRef("local_offset")),
#                            body=loop_body)
#
#             body.append(for_loop)
#             return MultiNode(body=body)
#
#     def transform(self, f, program_config):
#         subconfig, tuner = program_config
#         interior_space = subconfig['self'].interior_space
#         interior_size = reduce(operator.mul, interior_space, 1)
#         local_size = min(cl.clGetDeviceIDs()[-1].max_work_group_size, interior_size)
#
#         kernel_funcs = []
#         global_sizes = []
#         local_sizes = []
#         kernel_params = []
#         encode_func = None
#
#         f = super(OclGeneralizedSimpleMeshOpSpecializer, self).transform(f, program_config)[0]
#         f = include_mover(f)
#         # remove includes
#         while isinstance(f.body[0], CppInclude):
#             f.body.pop(0)
#         encode_func = f.body[0]
#
#         if local_size != interior_size:
#             first_reducer = f.find(FunctionDecl)
#             # need to change return statement to assign statement
#             first_reducer = DeclarationFiller().visit(first_reducer)
#             # remove return statement:
#             first_reducer.defn.pop()
#             first_reducer.defn.append(Assign(ArrayRef(SymbolRef("max_so_far"),
#                                                       FunctionCall(SymbolRef("get_global_id"), [Constant(0)])),
#                                              SymbolRef("max_norm")))
#             params = []
#             for param in first_reducer.params:
#                 if not isinstance(subconfig[param.name], (int, float, np.ndarray)):
#                     continue
#                 if isinstance(subconfig[param.name], (int, float)):
#                     param.type = ctypes.c_double()
#                 else:
#                     param.type = ctypes.POINTER(ctypes.c_double)()
#                     # param.set_restrict()
#                     param.set_global()
#                 params.append(param)
#             params.append(SymbolRef("max_so_far", ctypes.POINTER(ctypes.c_double)(), _global=True))
#             # params[-1].set_restrict()
#             first_reducer.params = params
#             kernel_funcs.append(first_reducer)
#             global_sizes.append(interior_size)
#             local_sizes.append(local_size)
#             kernel_params.append(first_reducer.params)
#
#         ### make the second reducer that looks like
#         second_reducer_defn = [
#             Assign(SymbolRef("max_norm", ctypes.c_double()), Constant(0.0)),
#             For(init=Assign(SymbolRef("i", ctypes.c_int()), Constant(0)),
#                 test=Lt(SymbolRef("i"), Constant(local_size)),
#                 incr=PostInc(SymbolRef("i")),
#                 body=[If(cond=Gt(ArrayRef(SymbolRef("max_so_far"), SymbolRef("i")), SymbolRef("max_norm")),
#                          then=Assign(SymbolRef("max_norm"), ArrayRef(SymbolRef("max_so_far"), SymbolRef("i")))),
#                       Assign(ArrayRef(SymbolRef("final"), Constant(0)), SymbolRef("max_norm"))])
#         ]
#
#         second_reducer_params = [
#             SymbolRef("max_so_far", ctypes.POINTER(ctypes.c_double)()),
#             SymbolRef("final", ctypes.POINTER(ctypes.c_double)())
#         ]
#         for param in second_reducer_params:
#             param.set_global()
#         #     param.set_restrict()
#
#         second_reducer = FunctionDecl(name="max_norm_2", params=second_reducer_params, defn=second_reducer_defn)
#         kernel_funcs.append(second_reducer)
#         global_sizes.append(local_size)
#         local_sizes.append(1)
#         kernel_params.append(second_reducer_params)
#         kernel_files = []
#         double_include = StringTemplate("""#pragma OPENCL EXTENSION cl_khr_fp64: enable""")
#
#         for kernel in kernel_funcs:
#             kernel.set_kernel()
#             for call in kernel.find_all(FunctionCall):
#                 if call.func.name == 'abs':
#                     call.func.name = 'fabs'
#                     # f.body.append(CppInclude("math.h"))
#             kernel_files.append(OclFile(name=kernel.name, body=[double_include, kernel]))
#
#
#         if len(kernel_files) > 1:
#             kernel_files[0].body.insert(0, encode_func)
#
#         control_file=generate_norm_mesh_control("norm_mesh", global_sizes, local_sizes, kernel_params, kernel_files)
#
#         files = [control_file]
#         files.extend(kernel_files)
#         # for file in files:
#         #     print(file)
#         return files
#
#     def finalize(self, transform_result, program_config):
#         for file in transform_result:
#             print(file)
#         subconfig, tuner_config = program_config
#         interior_space = subconfig['self'].interior_space
#         interior_size = reduce(operator.mul, interior_space, 1)
#         local_size = min(cl.clGetDeviceIDs()[-1].max_work_group_size, interior_size)
#
#         project = Project(transform_result)
#         control = transform_result[0]
#         kernels = transform_result[1:]
#         retval = ctypes.c_double
#         kernel_name = self.tree.body[0].name  # refers to original FunctionDef
#         entry_point = kernel_name + "_control"
#
#         param_types = [cl.cl_command_queue, cl.cl_kernel]
#         for param, value in subconfig.items():
#             if isinstance(subconfig[param], np.ndarray):
#                 param_types.append(cl.cl_mem)
#             elif isinstance(subconfig[param], (int, float)):
#                 param_types.append(ctypes.c_double)
#         param_types.append(cl.cl_mem)  # max_so_far
#         param_types.append(cl.cl_mem)  # final
#
#         level = subconfig['self']
#         kernels = [cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel.name] for kernel in kernels]
#         fn = MeshOpOclFunction(kernels, [np.ndarray((local_size,)), np.ndarray((1,))])
#         return fn.finalize(
#             entry_point, project, ctypes.CFUNCTYPE(retval, *param_types)
#         )
#
#
#
# def generate_norm_mesh_control(name, global_sizes, local_sizes, kernel_params, kernels):
#     # kernels are expected to be ocl files
#     defn = []
#     defn.append(ArrayRef(SymbolRef("global", ctypes.c_ulong()), Constant(1)))
#     defn.append(ArrayRef(SymbolRef("local", ctypes.c_ulong()), Constant(1)))
#     defn.append(Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0)))
#     for gsize, lsize, param_set, kernel in zip(global_sizes, local_sizes, kernel_params, kernels):
#         defn.append(Assign(ArrayRef(SymbolRef("global"), Constant(0)), Constant(gsize)))
#         defn.append(Assign(ArrayRef(SymbolRef("local"), Constant(0)), Constant(lsize)))
#         kernel_name = kernel.find(FunctionDecl, kernel=True).name
#         for param, num in zip(param_set, range(len(param_set))):
#             if isinstance(param, ctypes.POINTER(ctypes.c_double)):
#                 set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
#                                      [SymbolRef(kernel_name),
#                                       Constant(num),
#                                       FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
#                                       Ref(SymbolRef(param.name))])
#             else:
#                 set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
#                                      [SymbolRef(kernel_name),
#                                       Constant(num),
#                                       Constant(ctypes.sizeof(param.type)),
#                                       Ref(SymbolRef(param.name))])
#             defn.append(set_arg)
#         enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
#             SymbolRef("queue"),
#             SymbolRef(kernel_name),
#             Constant(1),
#             NULL(),
#             SymbolRef("global"),
#             SymbolRef("local"),
#             Constant(0),
#             NULL(),
#             NULL()
#         ])
#         defn.append(enqueue_call)
#         defn.append(FunctionCall(SymbolRef("clFinish"), [SymbolRef("queue")]))
#     defn.append(ArrayRef(SymbolRef("return_value", ctypes.c_double()), Constant(1)))
#     defn.append(FunctionCall(SymbolRef("clEnqueueReadBuffer"),
#                              [
#                                  SymbolRef("queue"),
#                                  SymbolRef("final"),
#                                  SymbolRef("CL_TRUE"),
#                                  Constant(0),
#                                  Constant(8),
#                                  Ref(SymbolRef("return_value")),
#                                  Constant(0),
#                                  NULL(),
#                                  NULL()
#                              ]))
#
#
#     defn.append(Return(ArrayRef(SymbolRef("return_value"), Constant(0))))
#     params=[]
#     params.append(SymbolRef("queue", cl.cl_command_queue()))
#     for kernel in kernels:
#         params.append(SymbolRef(kernel.find(FunctionDecl, kernel=True).name, cl.cl_kernel()))
#
#     params.append(SymbolRef("mesh", cl.cl_mem()))
#     params.append(SymbolRef("max_so_far", cl.cl_mem()))
#     params.append(SymbolRef("final", cl.cl_mem()))
#
#     control_func = FunctionDecl(return_type=ctypes.c_double(), name="%s_control"%name, params=params, defn=defn)
#     ocl_include = StringTemplate("""
#             #include <stdio.h>
#             #ifdef __APPLE__
#             #include <OpenCL/opencl.h>
#             #else
#             #include <CL/cl.h>
#             #endif
#             """)
#     return CFile("%s_control"%name, body=[ocl_include, control_func], config_target="opencl")
