import ast
from collections import OrderedDict
import ctypes
from ctree.c.macros import NULL
from ctree.c.nodes import SymbolRef, Constant, PostInc, Lt, For, Assign, FunctionDecl, CFile, Return, FunctionCall, \
    MultiNode, ArrayRef, Array, Ref, Mul, Add, If, Gt, AddAssign, Div, Eq, ArrayDef
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction
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

    def set_dirty_buffers(self, args):
        args[1].buffer.dirty = True


class MeshReduceOpOclFunction(PyGMGOclConcreteSpecializedFunction):

    def get_all_args(self, args, kwargs):
        return args + self.extra_args

    def reduced_value(self):
        ary, evt = cl.buffer_to_ndarray(self.queue, self.extra_args[-1].buffer.buffer, self.extra_args[-1])
        evt.wait()
        return self.extra_args[-1][0]


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
            CRangeTransformer(),
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



class MeshOpOclSpecializer(LazySpecializedFunction):

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

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        ndim = subconfig['self'].solver.dimensions
        layers = [
            ParamStripper(('self')),
            SemanticFinder(subconfig),
            self.RangeTransformer(),
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


class OclFillMeshSpecializer(MeshOpOclSpecializer):

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
        global_size = reduce(operator.mul, global_shape, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        while global_size % local_size != 0:
            local_size -= 1
        control = new_generate_control("%s_control" % func_name, global_size, local_size, func_decl.params, [func_decl])
        # print(control)
        # raise TypeError
        return [control, kernel]
        # return [kernel]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        project = Project(transform_result)
        kernel = transform_result[1]
        level = subconfig['self']

        global_shape = tuple(dim + 2 * ghost for dim, ghost in zip(level.interior_space, level.ghost_zone))
        global_size = reduce(operator.mul, global_shape, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        while global_size % local_size != 0:
            local_size -= 1

        retval = ctypes.c_int
        kernel_name = self.tree.body[0].name
        entry_point = kernel_name + "_control"
        # param_types = [cl.cl_command_queue, cl.cl_kernel]
        param_types = []
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        entry_type = [ctypes.c_int32, cl.cl_command_queue, cl.cl_kernel]
        entry_type.extend(param_types)
        level = subconfig['self']
        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel_name + "_kernel"]
        kernel.argtypes = param_types
        kernel = KernelRunManager(kernel, global_size, local_size)
        fn = MeshOpOclFunction()

        return fn.finalize(
            entry_point, project, ctypes.CFUNCTYPE(*entry_type),
            level, [kernel]
        )
        # return fn.finalize(project, level, [kernel])


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


class OclGeneralizedSimpleMeshOpSpecializer(MeshOpOclSpecializer):

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
        # global_shape = tuple(dim + 2 * ghost for dim, ghost in zip(subconfig['self'].interior_space, subconfig['self'].ghost_zone))
        global_size = reduce(operator.mul, subconfig['self'].interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        while global_size % local_size != 0:
            local_size -= 1
        control = new_generate_control("%s_control" % func_name, global_size, local_size, func_decl.params, [func_decl])
        # print(control)
        # raise TypeError
        return [control, kernel]
        # return [kernel]

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
        # param_types = []
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[kernel_name + "_kernel"]
        kernel.argtypes = param_types[2:]
        kernel = KernelRunManager(kernel, global_size, local_size)
        fn = MeshOpOclFunction()
        return fn.finalize(
            entry_point, project, ctypes.CFUNCTYPE(retval, *param_types),
            level, [kernel]
        )
        # return fn.finalize(project, level, [kernel])


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

        f = super(OclGeneralizedSimpleMeshOpSpecializer, self).transform(f, program_config)[0]
        f = include_mover(f)
        # remove includes
        while isinstance(f.body[0], CppInclude):
            f.body.pop(0)
        encode_func = f.body[0]

        kernel = f.find(FunctionDecl)
        kernel = DeclarationFiller().visit(kernel)
        acc_name = get_reduction_var_name(kernel.name)

        if kernel.name == "norm_mesh":
            for_loop = kernel.find(For)
            if_statement = for_loop.body.pop()
            temp_assign = if_statement.then[0].body[0]
            max_assign = Assign(SymbolRef("max_norm"),
                                   FunctionCall(SymbolRef("fmax"),
                                                [SymbolRef("max_norm"), SymbolRef("____temp__max_norm")]))
            for_loop.body.append(temp_assign)
            for_loop.body.append(max_assign)

        # replace return statement with assignment
        kernel.defn[-1] = Assign(ArrayRef(SymbolRef("temp"), FunctionCall(SymbolRef("get_global_id"), [Constant(0)])),
                                 SymbolRef(acc_name))

        kernel.defn.append(StringTemplate("""barrier(CLK_GLOBAL_MEM_FENCE);"""))

        kernel.defn.append(If(cond=Eq(SymbolRef("global_offset"), Constant(0)),
                              then=generate_final_reduction(kernel.name, interior_size, local_size)))

        params = []
        for param in kernel.params:
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
        params.append(SymbolRef("final", ctypes.POINTER(ctypes.c_double)(), _global=True))
        # params[-1].set_restrict()

        kernel.params = params
        kernel.set_kernel()
        for call in kernel.find_all(FunctionCall):
            if call.func.name == 'abs':
                call.func.name = 'fabs'

        ocl_file = OclFileWrapper(kernel.name).visit(CFile(body=[encode_func, kernel]))
        control_file = generate_reducer_control("%s_control" % kernel.name,
                                                local_size, local_size, kernel.params, [kernel])
        files = [control_file, ocl_file]
        # print(control_file)
        # raise TypeError
        return files
        # return [ocl_file]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        level = subconfig['self']

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

        project = Project(transform_result)
        control = transform_result[0]
        kernel = transform_result[1]
        retval = ctypes.c_double
        kernel_name = self.tree.body[0].name
        entry_point = kernel_name + "_control"

        param_types = [cl.cl_command_queue, cl.cl_kernel]
        # param_types = []
        for param, value in subconfig.items():
            if isinstance(subconfig[param], np.ndarray):
                param_types.append(cl.cl_mem)
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)
        param_types.append(cl.cl_mem)  # temp
        param_types.append(cl.cl_mem)  # final

        name = kernel.name

        kernel = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()[name]
        kernel.argtypes = param_types[2:]
        kernel = KernelRunManager(kernel, local_size, local_size)
        # kernels[0].argtypes = tuple(param_types[len(kernels)+1:-1])
        # kernels[1].argtypes = tuple(param_types[-2:])
        # kernels[0] = KernelRunManager(kernels[0], local_size, local_size)
        # kernels[1] = KernelRunManager(kernels[1], local_size, 1)
        extra_args = (temp_mesh, final_mesh)
        fn = MeshReduceOpOclFunction()
        fn = fn.finalize(entry_point, project, ctypes.CFUNCTYPE(retval, *param_types),
                         level, [kernel], extra_args)
        # fn = fn.finalize(project, level, [kernel], extra_args)
        return fn


def get_reduction_var_name(func_name):
    if func_name == "norm_mesh":
        return "max_norm"
    if func_name == "dot_mesh" or func_name == "mean_mesh":
        return "accumulator"

def generate_final_reduction(func_name, interior_size, local_size):
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
    defn.append(Assign(SymbolRef(acc_name), Constant(0.0)))
    for_loop = For(init=Assign(SymbolRef("i", ctypes.c_int()), Constant(0)),
                   test=Lt(SymbolRef("i"), Constant(local_size)),
                   incr=PostInc(SymbolRef("i")),
                   body=loop_body)
    defn.append(for_loop)
    defn.append(final_assignment)

    return MultiNode(body=defn)


def generate_reducer_control(name, global_size, local_size, kernel_params, kernels):
    defn = []
    defn.append(ArrayDef(SymbolRef("global", ctypes.c_ulong()), 1, Array(body=[Constant(global_size)])))
    defn.append(ArrayDef(SymbolRef("local", ctypes.c_ulong()), 1, Array(body=[Constant(local_size)])))
    for kernel in kernels:
        kernel_name = kernel.find(FunctionDecl, kernel=True).name
        for param, num in zip(kernel_params, range(len(kernel_params))):
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
    defn.append(ArrayRef(SymbolRef("return_value", ctypes.c_double()), Constant(1)))
    defn.append(StringTemplate("""clFinish(queue);"""))
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
    for param in kernel_params:
        if isinstance(param.type, ctypes.POINTER(ctypes.c_double)):
            params.append(SymbolRef(param.name, cl.cl_mem()))
        else:
            params.append(param)
    func = FunctionDecl(ctypes.c_int(), name, params, defn)
    ocl_include = StringTemplate("""
            #include <stdio.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
    body = [ocl_include, func]
    file = CFile(name=name, body=body, config_target='opencl')
    return file
