import ast
from collections import OrderedDict
import ctypes
from ctree.c.macros import NULL
from ctree.c.nodes import SymbolRef, Constant, PostInc, Lt, For, Assign, FunctionDecl, CFile, Return, FunctionCall, \
    MultiNode, ArrayRef, Array, Ref, Mul, Add, If, Gt, AddAssign, Div
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
from hpgmg.finite_volume.mesh import Mesh, Buffer
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, KernelRunManager, \
    PyGMGOclConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.smooth_specializer import apply_all_layers
from hpgmg.finite_volume.operators.specializers.util import include_mover, flattened_to_multi_index, \
    new_generate_control, compute_largest_local_work_size
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops

from ctree.frontend import dump
import numpy as np
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexDirectTransformer, \
    IndexTransformer, OclFileWrapper
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


class MeshOpOclFunction(PyGMGOclConcreteSpecializedFunction):

    # def set_kernel_args(self, args, kwargs):
    #     # need to set kernel argtypes!!!
    #     kernel = self.kernels[0]  # there is one kernel until we do reductions
    #
    #     kernel_args = []
    #     # kernel_argtypes = []
    #
    #     for arg in args:
    #         if isinstance(arg, Mesh):
    #             mesh = arg
    #             if mesh.dirty:
    #                 buffer = None if mesh.buffer is None else mesh.buffer.buffer
    #                 buf, evt = cl.buffer_from_ndarray(self.queue, mesh, buf=buffer)
    #                 mesh.buffer = buf
    #                 mesh.buffer.evt = evt
    #                 mesh.dirty = False
    #
    #             elif mesh.buffer is None:
    #                 size = mesh.size * ctypes.sizeof(ctypes.c_double)
    #                 mesh.buffer = cl.clCreateBuffer(self.context, size)
    #
    #             kernel_args.append(mesh.buffer)
    #             # kernel_argtypes.append(cl.cl_mem)
    #
    #         elif isinstance(arg, (int, float)):
    #             kernel_args.append(arg)
    #             # kernel_argtypes.append(ctypes.c_double) # ?????
    #
    #     kernel.args = kernel_args
    #     # kernel.kernel.argtypes = tuple(kernel_argtypes)


    def __call__(self, *args, **kwargs):
        self.set_kernel_args(args, kwargs)

        kernel = self.kernels[0]
        kernel_args = []
        previous_events = []
        for arg in kernel.args:
            if isinstance(arg, Buffer):
                kernel_args.append(arg.buffer)
                if arg.evt is not None:
                    previous_events.append(arg.evt)
            else:
                kernel_args.append(arg)

        cl.clWaitForEvents(*previous_events)

        run_evt = kernel.kernel(*kernel_args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
        # run_evt.wait()
        #
        # ary, evt = cl.buffer_to_ndarray(self.queue, kernel.args[0].buffer, args[1])
        # kernel.args[0].evt = evt
        # kernel.args[0].dirty = False
        # kernel.args[0].evt.wait()

        args[1].buffer.evt = run_evt
        args[1].buffer.dirty = True


class MeshReduceOpOclFunction(PyGMGOclConcreteSpecializedFunction):

    def set_kernel_args(self, args, kwargs):
        bufferized_args = []

        for arg in args:
            if isinstance(arg, Mesh):
                mesh = arg
                if mesh.dirty:
                    buffer = None if mesh.buffer is None else mesh.buffer.buffer
                    buf, evt = cl.buffer_from_ndarray(self.queue, mesh, buf=buffer)
                    mesh.buffer = buf
                    mesh.buffer.evt = evt
                    mesh.dirty = False

                elif mesh.buffer is None:
                    size = mesh.size * ctypes.sizeof(ctypes.c_double)
                    mesh.buffer = cl.clCreateBuffer(self.context, size)

                bufferized_args.append(mesh.buffer)

            elif isinstance(arg, (int, float)):
                bufferized_args.append(arg)

        self.kernels[0].args = bufferized_args[:-1]
        self.kernels[1].args = bufferized_args[-2:]


    def __call__(self, *args, **kwargs):
        args = args + self.extra_args
        self.set_kernel_args(args, kwargs)

        for kernel in self.kernels:
            kernel_args = []
            previous_events = []
            for arg in kernel.args:
                if isinstance(arg, Buffer):
                    kernel_args.append(arg.buffer)
                    if arg.evt:
                        previous_events.append(arg.evt)
                else:
                    kernel_args.append(arg)

            cl.clWaitForEvents(*previous_events)
            run_evt = kernel.kernel(*kernel_args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
            run_evt.wait()

        ary, evt = cl.buffer_to_ndarray(self.queue, args[-1].buffer.buffer, args[-1])
        evt.wait()
        return args[-1][0]

# class MeshOpOclFunction(ConcreteSpecializedFunction):
#
#     def __init__(self, kernels, other_args=None):
#         self.kernels = kernels
#         self.other_args = other_args if other_args else []
#
#     def finalize(self, entry_point_name, project_node, entry_point_typesig):
#         self.entry_point_name = entry_point_name
#         self.entry_point_typesig = entry_point_typesig
#         self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
#         return self
#
#     # def set_kernel_args(self, level, args, kwargs):
#     #     # args contains the extra args already
#     #     # we know that there are two kernels
#     #     # and that the arguments to the second kernels are just temp and final
#     #     queue = level.queue
#     #     bufferized_args = []
#     #     for arg in args:
#     #         if isinstance(arg, Mesh):
#     #             mesh = arg
#     #             if mesh.buffer and mesh.buffer.dirty:
#     #                 mesh.buffer, evt = cl.buffer_from_ndarray(queue, mesh, mesh.buffer)
#     #                 mesh.buffer.evt = evt
#     #             elif mesh.dirty:
#     #                 mesh.buffer, evt = cl.buffer_from_ndarray(queue, mesh)
#     #                 mesh.buffer.evt = evt
#     #             else:
#     #                 size = mesh.size * ctypes.sizeof(ctypes.c_double)
#     #                 mesh.buffer = cl.clCreateBuffer(level.context, size)
#     #                 mesh.buffer.evt = None
#     #             mesh.buffer.dirty = False
#     #             bufferized_args.append(mesh.buffer)
#     #         elif isinstance(arg, (int, float)):
#     #             bufferized_args.append(arg)
#     #
#     #     self.kernels[0].args = bufferized_args[:-1]
#     #     self.kernels[1].args = bufferized_args[-2:]
#
#     def __call__(self, *args, **kwargs):
#         arguments = []
#         queue = None
#         args = args + tuple(self.other_args)
#         for arg in args:
#             if hasattr(arg, "queue"):
#                 queue = arg.queue
#                 arguments.append(queue)
#                 arguments.extend(kernel for kernel in self.kernels)
#             elif isinstance(arg, Mesh) or isinstance(arg, np.ndarray):
#                 if hasattr(arg, "buffer"):
#                     if arg.buffer is None:
#                         arg.buffer, evt = cl.buffer_from_ndarray(queue, arg)
#                     else:
#                         arg.buffer, evt = cl.buffer_from_ndarray(queue, arg, buf=arg.buffer)
#                     arguments.append(arg.buffer)
#                 else:
#                     buf, evt = cl.buffer_from_ndarray(queue, arg)
#                     arguments.append(buf)
#             elif isinstance(arg, (int, float)):
#                 arguments.append(arg)
#         #
#         # value = self._c_function(*arguments)
#         # for arg in args:
#         #     if isinstance(arg, Mesh):
#         #         arg, evt = cl.buffer_to_ndarray(queue, arg.buffer, out=arg)
#         #
#         # return value


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
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
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
        level = subconfig['self']

        global_shape = tuple(dim + 2 * ghost for dim, ghost in zip(level.interior_space, level.ghost_zone))
        global_size = reduce(operator.mul, global_shape, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        while global_size % local_size != 0:
            local_size -= 1

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
        kernel.argtypes = tuple(param_types[2:])
        kernel = KernelRunManager(kernel, global_size, local_size)
        fn = MeshOpOclFunction()
        return fn.finalize(
            entry_point, project, ctypes.CFUNCTYPE(retval, *param_types),
            level.context, level.queue, [kernel]
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
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        while global_size % local_size != 0:
            local_size -= 1
        control = new_generate_control("%s_control" % func_name, global_size, local_size, func_decl.params, [func_decl])
        # print(control)
        # print(kernel)
        return [control, kernel]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        level = subconfig['self']
        project = Project(transform_result)
        control = transform_result[0]
        kernel = transform_result[1]

        global_size = reduce(operator.mul, level.interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        while global_size % local_size != 0:
            local_size -= 1

        retval = ctypes.c_int
        kernel_name = self.tree.body[0].name  # refers to original FunctionDef
        entry_point = kernel_name + "_control"
        param_types = [cl.cl_command_queue, cl.cl_kernel]
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel_name + "_kernel"]
        kernel.argtypes = tuple(param_types[2:])
        kernel = KernelRunManager(kernel, global_size, local_size)
        fn = MeshOpOclFunction()
        return fn.finalize(
            entry_point, project, ctypes.CFUNCTYPE(retval, *param_types),
            level.context, level.queue, [kernel]
        )


class OclMeshReduceOpSpecializer(OclGeneralizedSimpleMeshOpSpecializer):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):

            ranges = node.iterator.ranges
            offsets = tuple(r[0] for r in ranges)
            shape = tuple(r[1] - r[0] for r in ranges)
            ndim = len(shape)
            global_size = reduce(operator.mul, shape, 1)  # this is an interior space
            local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
            if global_size > cl.clGetDeviceIDs()[-1].max_work_group_size:
                stride = global_size / cl.clGetDeviceIDs()[-1].max_work_group_size
            else:
                stride = 1

            body = []
            global_id = FunctionCall(SymbolRef("get_global_id"), [Constant(0)])
            body.append(Assign(SymbolRef("global_offset", ctypes.c_int()), Mul(global_id, Constant(stride))))
            body.append(SymbolRef("____temp__max_norm", ctypes.c_double()))

            body.extend(SymbolRef("index_%d"%d, ctypes.c_int()) for d in range(ndim))

            indices = flattened_to_multi_index(SymbolRef("global_id"),
                                               shape, offsets=offsets)

            loop_body = []
            loop_body.extend(Assign(SymbolRef("index_%d"%d), indices[d]) for d in range(ndim))
            loop_body.extend(node.body)

            for_loop = For(init=Assign(SymbolRef("global_id", ctypes.c_int()), global_id),
                           test=Lt(SymbolRef("global_id"), Constant(global_size)),
                           incr=AddAssign(SymbolRef("global_id"), Constant(local_size)),
                           body=loop_body)

            body.append(for_loop)
            return MultiNode(body=body)

    def transform(self, f, program_config):
        subconfig, tuner = program_config
        interior_space = subconfig['self'].interior_space
        interior_size = reduce(operator.mul, interior_space, 1)
        local_size = min(cl.clGetDeviceIDs()[-1].max_work_group_size, interior_size)

        kernel_funcs = []

        f = super(OclGeneralizedSimpleMeshOpSpecializer, self).transform(f, program_config)[0]
        f = include_mover(f)
        # remove includes
        while isinstance(f.body[0], CppInclude):
            f.body.pop(0)
        encode_func = f.body[0]

        #always launching two kernels no matter what, for now at least
        # if local_size != interior_size:

        first_reducer = f.find(FunctionDecl)
        # need to change return statement to assign statement
        first_reducer = DeclarationFiller().visit(first_reducer)
        acc_name = get_reduction_var_name(first_reducer.name)

        # # if we are doing norm_mesh, need to remove if statement
        if first_reducer.name == "norm_mesh":
            for_loop = first_reducer.find(For)
            if_statement = for_loop.body.pop()
            temp_assign = if_statement.then[0].body[0]
            max_assign = Assign(SymbolRef("max_norm"),
                                   FunctionCall(SymbolRef("fmax"),
                                                [SymbolRef("max_norm"), SymbolRef("____temp__max_norm")]))
            for_loop.body.append(temp_assign)
            for_loop.body.append(max_assign)

        # remove return statement and add assignment statement:
        first_reducer.defn[-1] = Assign(ArrayRef(SymbolRef("temp"),
                                                  FunctionCall(SymbolRef("get_global_id"), [Constant(0)])),
                                         SymbolRef(acc_name))

        # params for first_reducer
        params = []
        for param in first_reducer.params:
            if not isinstance(subconfig[param.name], (int, float, np.ndarray)):
                continue
            if isinstance(subconfig[param.name], (int, float)):
                param.type = ctypes.c_double()
            else:
                param.type = ctypes.POINTER(ctypes.c_double)()
                # param.set_restrict()
                param.set_global()
            params.append(param)
        params.append(SymbolRef("temp", ctypes.POINTER(ctypes.c_double)(), _global=True))
        # params[-1].set_restrict()

        first_reducer.params = params
        kernel_funcs.append(first_reducer)

        # make the second reducer
        second_reducer_defn = generate_second_reducer_func(first_reducer.name, interior_size, local_size)

        second_reducer_params = [
            SymbolRef("temp", ctypes.POINTER(ctypes.c_double)(), _global=True),
            SymbolRef("final", ctypes.POINTER(ctypes.c_double)(), _global=True)
        ]
        for param in second_reducer_params:
            param.set_global()
        #     param.set_restrict()

        second_reducer = FunctionDecl(name="%s_2"%first_reducer.name, params=second_reducer_params, defn=second_reducer_defn)
        kernel_funcs.append(second_reducer)
        kernel_files = []
        double_include = StringTemplate("""#pragma OPENCL EXTENSION cl_khr_fp64: enable""")

        for kernel in kernel_funcs:
            kernel.set_kernel()
            for call in kernel.find_all(FunctionCall):
                if call.func.name == 'abs':
                    call.func.name = 'fabs'
            kernel_files.append(OclFile(name=kernel.name, body=[double_include, kernel]))
        kernel_files[0].body.insert(0, encode_func)

        control_file=generate_reduce_mesh_control(first_reducer.name,
                                                [interior_size, local_size],
                                                [local_size, 1],
                                                first_reducer.params[:-1],
                                                kernel_files)

        files = [control_file]
        files.extend(kernel_files)
        return files

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        level = subconfig['self']
        # interior_size = reduce(operator.mul, interior_space, 1)

        global_size = reduce(operator.mul, level.interior_space, 1)
        local_size = min(cl.clGetDeviceIDs()[-1].max_work_group_size, global_size)
        while global_size % local_size != 0:
            local_size -= 1

        if level.interior_space not in level.reducer_meshes:
            level.reducer_meshes[level.interior_space] = Mesh(global_size)
        if (1,) not in level.reducer_meshes:
            level.reducer_meshes[(1,)] = Mesh((1,))
        temp_mesh = level.reducer_meshes[level.interior_space]
        final_mesh = level.reducer_meshes[(1,)]

        # print("I AM USING THE LOCAL SIZE {} FOR A LEVEL OF SHAPE {}".format(local_size, level.interior_space))

        project = Project(transform_result)
        control = transform_result[0]
        kernels = transform_result[1:]
        retval = ctypes.c_double
        kernel_name = self.tree.body[0].name  # refers to original FunctionDef
        entry_point = kernel_name + "_control"

        param_types = [cl.cl_command_queue]
        param_types.extend(cl.cl_kernel for _ in range(len(kernels)))
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        param_types.append(cl.cl_mem)  # temp
        param_types.append(cl.cl_mem)  # final

        kernels = [cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel.name] for kernel in kernels]
        kernels[0].argtypes = tuple(param_types[len(kernels)+1:-1])
        kernels[1].argtypes = tuple(param_types[-2:])
        kernels[0] = KernelRunManager(kernels[0], local_size, local_size)
        kernels[1] = KernelRunManager(kernels[1], local_size, 1)
        extra_args = (temp_mesh, final_mesh)
        fn = MeshReduceOpOclFunction()
        fn = fn.finalize(entry_point, project, ctypes.CFUNCTYPE(retval, *param_types),
                         level.context, level.queue, kernels, extra_args)
        return fn



def generate_reduce_mesh_control(name, global_sizes, local_sizes, original_params, kernels):
    # kernels are expected to be ocl files
    first_reducer_params = original_params[:]
    first_reducer_params.append(SymbolRef("temp", ctypes.POINTER(ctypes.c_double)()))
    second_reducer_params = [
        SymbolRef("temp", ctypes.POINTER(ctypes.c_double)()),
        SymbolRef("final", ctypes.POINTER(ctypes.c_double)())
    ]
    kernel_params = [first_reducer_params, second_reducer_params]
    defn = []
    defn.append(ArrayRef(SymbolRef("global", ctypes.c_ulong()), Constant(1)))
    defn.append(ArrayRef(SymbolRef("local", ctypes.c_ulong()), Constant(1)))
    defn.append(Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0)))
    for gsize, lsize, param_set, kernel in zip(global_sizes, local_sizes, kernel_params, kernels):
        defn.append(Assign(ArrayRef(SymbolRef("global"), Constant(0)), Constant(gsize)))
        defn.append(Assign(ArrayRef(SymbolRef("local"), Constant(0)), Constant(lsize)))
        kernel_name = kernel.find(FunctionDecl, kernel=True).name
        for param, num in zip(param_set, range(len(param_set))):
            if isinstance(param, ctypes.POINTER(ctypes.c_double)):
                set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                     [SymbolRef(kernel_name),
                                      Constant(num),
                                      FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
                                      Ref(SymbolRef(param.name))])
            else:
                set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                     [SymbolRef(kernel_name),
                                      Constant(num),
                                      Constant(ctypes.sizeof(param.type)),
                                      Ref(SymbolRef(param.name))])
            defn.append(set_arg)
        enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
            SymbolRef("queue"),
            SymbolRef(kernel_name),
            Constant(1),
            NULL(),
            SymbolRef("global"),
            SymbolRef("local"),
            Constant(0),
            NULL(),
            NULL()
        ])
        defn.append(enqueue_call)
        defn.append(FunctionCall(SymbolRef("clFinish"), [SymbolRef("queue")]))
    defn.append(ArrayRef(SymbolRef("return_value", ctypes.c_double()), Constant(1)))
    defn.append(FunctionCall(SymbolRef("clEnqueueReadBuffer"),
                             [
                                 SymbolRef("queue"),
                                 SymbolRef("final"),
                                 SymbolRef("CL_TRUE"),
                                 Constant(0),
                                 Constant(8),
                                 Ref(SymbolRef("return_value")),
                                 Constant(0),
                                 NULL(),
                                 NULL()
                             ]))


    defn.append(Return(ArrayRef(SymbolRef("return_value"), Constant(0))))
    params=[]
    params.append(SymbolRef("queue", cl.cl_command_queue()))
    for kernel in kernels:
        params.append(SymbolRef(kernel.find(FunctionDecl, kernel=True).name, cl.cl_kernel()))

    # params.append(SymbolRef("mesh", cl.cl_mem()))
    for param in original_params:
        if isinstance(param.type, ctypes.POINTER(ctypes.c_double)):
            params.append(SymbolRef(param.name, cl.cl_mem()))
        else:
            params.append(param)
    params.append(SymbolRef("temp", cl.cl_mem()))
    params.append(SymbolRef("final", cl.cl_mem()))

    control_func = FunctionDecl(return_type=ctypes.c_double(), name="%s_control"%name, params=params, defn=defn)
    ocl_include = StringTemplate("""
            #include <stdio.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
    return CFile("%s_control"%name, body=[ocl_include, control_func], config_target="opencl")

def get_reduction_var_name(func_name):
    if func_name == "norm_mesh":
        return "max_norm"
    if func_name == "dot_mesh" or func_name == "mean_mesh":
        return "accumulator"

def generate_second_reducer_func(func_name, interior_size, local_size):
    loop_body = []
    final_assignment = None
    acc_name = get_reduction_var_name(func_name)
    if func_name == "norm_mesh":
        loop_body.append(If(cond=Gt(ArrayRef(SymbolRef("temp"), SymbolRef("i")), SymbolRef(acc_name)),
                   then=Assign(SymbolRef(acc_name), ArrayRef(SymbolRef("temp"), SymbolRef("i")))))
        final_assignment = Assign(ArrayRef(SymbolRef("final"), Constant(0)), SymbolRef(acc_name))
    elif func_name == "dot_mesh":
        loop_body.append(AddAssign(SymbolRef(acc_name), ArrayRef(SymbolRef("temp"), SymbolRef("i"))))
        final_assignment = Assign(ArrayRef(SymbolRef("final"), Constant(0)), SymbolRef(acc_name))
    elif func_name == "mean_mesh":
        loop_body.append(AddAssign(SymbolRef(acc_name), ArrayRef(SymbolRef("temp"), SymbolRef("i"))))
        final_assignment = Assign(ArrayRef(SymbolRef("final"), Constant(0)),
                                  Div(SymbolRef(acc_name), Constant(interior_size)))

    defn = []
    defn.append(Assign(SymbolRef(acc_name, ctypes.c_double()), Constant(0.0)))
    for_loop = For(init=Assign(SymbolRef("i", ctypes.c_int()), Constant(0)),
                   test=Lt(SymbolRef("i"), Constant(local_size)),
                   incr=PostInc(SymbolRef("i")),
                   body=loop_body)
    defn.append(for_loop)
    defn.append(final_assignment)

    return defn
