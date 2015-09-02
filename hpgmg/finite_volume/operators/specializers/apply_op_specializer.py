from ctree.c.nodes import FunctionCall, SymbolRef, Assign, Constant, MultiNode, Add, ArrayDef, Array, CFile
from ctree.jit import LazySpecializedFunction
from ctree.nodes import Project
from ctree.ocl.nodes import OclFile
from ctree.transformations import PyBasicConversions
from ctree.transforms.declaration_filler import DeclarationFiller
from hpgmg.finite_volume.operators.specializers.jit import PyGMGOclConcreteSpecializedFunction, KernelRunManager, \
    PyGMGConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, compute_local_work_size, \
    flattened_to_multi_index, to_macro_function, new_generate_control, include_mover
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer, OclFileWrapper
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from ast import Name
import ast
import math
import pycl as cl
import ctypes
import operator
import numpy as np

__author__ = 'dorthyluu'

class CApplyOpFunction(PyGMGConcreteSpecializedFunction):
    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        level, target_mesh, source_mesh = args
        c_args = [target_mesh, source_mesh] + level.beta_face_values + [level.alpha]
        return c_args, {}


class OclApplyOpFunction(PyGMGOclConcreteSpecializedFunction):

    def get_all_args(self, args, kwargs):
        [level, target_mesh, source_mesh] = args
        args_to_bufferize = [target_mesh, source_mesh] + level.beta_face_values + [level.alpha]
        return args_to_bufferize

    def set_dirty_buffers(self, args):
        args[1].buffer.dirty = True


class CApplyOpSpecializer(LazySpecializedFunction):

    class ApplyOpSubconfig(dict):
        def __hash__(self):
            operator = self['level'].solver.problem_operator
            hashed = [
                operator.a, operator.b, operator.h2inv,
                operator.dimensions, operator.is_variable_coefficient,
                operator.ghost_zone, tuple(operator.neighborhood_offsets)
            ]
            for i in ('target_mesh', 'source_mesh'):
                hashed.append(self[i].shape)
            return hash(tuple(hashed))

    def args_to_subconfig(self, args):
        params = ('level', 'target_mesh', 'source_mesh')
        return self.ApplyOpSubconfig({
            param: arg for param, arg in zip(params, args)
        })

    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['level'].solver.problem_operator.dimensions
        # shape = subconfig['level'].interior_space
        # global_size = reduce(operator.mul, shape, 1)
        # ghost_zone = subconfig['level'].solver.problem_operator.ghost_zone
        #
        # device = cl.clGetDeviceIDs()[-1]
        # local_size = compute_local_work_size(device, shape)
        # single_work_dim = int(round(local_size ** (1/float(ndim))))
        # local_work_shape = tuple(single_work_dim for _ in range(ndim))
        #
        layers = [
            ParamStripper(('level')),
            AttributeRenamer({'level.solver.problem_operator.apply_op': Name('apply_op', ast.Load())}),
            SemanticFinder(subconfig),
            CRangeTransformer(),
            AttributeGetter({'level': subconfig['level']}),
            ArrayRefIndexTransformer(
                encode_map={'index': 'encode'},
                ndim=ndim),
            PyBasicConversions(),
        ]
        func = apply_all_layers(layers, func)

        defn = []

        beta_def = ArrayDef(
            SymbolRef("beta_face_values", sym_type=ctypes.POINTER(ctypes.c_double)()),
            size=ndim,
            body=Array(
                body=[
                    SymbolRef("beta_face_values_{}".format(i)) for i in range(ndim)
                ]
            )
        )

        defn.append(beta_def)
        defn.extend(func.defn)
        func.defn = defn
        for call in func.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()  #remove level

        macro_func = to_macro_function(subconfig['level'].solver.problem_operator.apply_op,
                                       rename={"level.beta_face_values": SymbolRef("beta_face_values"),
                                               "level.alpha": SymbolRef("alpha")})

        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)

        params = func.params
        params.extend([
            SymbolRef("beta_face_values_{}".format(i), sym_type=ctypes.POINTER(ctypes.c_double)())
            for i in range(ndim)
        ])
        params.append(
            SymbolRef("alpha", sym_type=ctypes.POINTER(ctypes.c_double)())
        )

        for param in params:
            param.type = ctypes.POINTER(ctypes.c_double)()

        cfile = CFile(body=[encode_func, macro_func, func])
        cfile = include_mover(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        subconfig, tuner = program_config
        param_types = [np.ctypeslib.ndpointer(thing.dtype, len(thing.shape), thing.shape) for thing in
        [
            subconfig[key] for key in ('target_mesh', 'source_mesh')
        ]]
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_type = np.ctypeslib.ndpointer(
            beta_sample.dtype,
            len(beta_sample.shape),
            beta_sample.shape
        )
        param_types.extend(
            [beta_type]*subconfig['level'].solver.problem_operator.dimensions
        )
        param_types.append(
            param_types[-1]
        )  # add 1 more for alpha
        fn = CApplyOpFunction()
        return fn.finalize(self.tree.body[0].name, Project(transform_result), ctypes.CFUNCTYPE(None, *param_types))


class OclApplyOpSpecializer(LazySpecializedFunction):

    class RangeTransformer(ast.NodeTransformer):

        def __init__(self, shape, ghost_zone, local_work_shape):
            self.shape = shape
            self.ghost_zone = ghost_zone
            self.local_work_shape = local_work_shape

        def visit_RangeNode(self, node):
            ndim = len(self.shape)
            global_work_dims = tuple(global_dim // work_dim for
                                     global_dim, work_dim in zip(self.shape, self.local_work_shape))

            body = []
            body.append(Assign(SymbolRef("group_id", ctypes.c_int()),
                               FunctionCall(SymbolRef("get_group_id"), [Constant(0)])))
            body.append(Assign(SymbolRef("local_id", ctypes.c_int()),
                               FunctionCall(SymbolRef("get_local_id"), [Constant(0)])))

            global_indices = flattened_to_multi_index(SymbolRef("group_id"), global_work_dims,
                                                      self.local_work_shape, self.ghost_zone)
            local_indices = flattened_to_multi_index(SymbolRef("local_id"), self.local_work_shape)
            for d in range(ndim):
                body.append(Assign(SymbolRef("index_%d"%d, ctypes.c_int()), Add(global_indices[d], local_indices[d])))

            body.extend(node.body)
            return MultiNode(body=body)

    class ApplyOpSubconfig(dict):
        def __hash__(self):
            operator = self['level'].solver.problem_operator
            hashed = [
                operator.a, operator.b, operator.h2inv,
                operator.dimensions, operator.is_variable_coefficient,
                operator.ghost_zone, tuple(operator.neighborhood_offsets)
            ]
            for i in ('target_mesh', 'source_mesh'):
                hashed.append(self[i].shape)
            return hash(tuple(hashed))

    def args_to_subconfig(self, args):
        params = ('level', 'target_mesh', 'source_mesh')
        return self.ApplyOpSubconfig({
            param: arg for param, arg in zip(params, args)
        })

    def transform(self, tree, program_config):
        kernel = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['level'].solver.problem_operator.dimensions
        shape = subconfig['level'].interior_space
        global_size = reduce(operator.mul, shape, 1)
        ghost_zone = subconfig['level'].solver.problem_operator.ghost_zone

        device = cl.clGetDeviceIDs()[-1]
        local_size = compute_local_work_size(device, shape)
        single_work_dim = int(round(local_size ** (1/float(ndim))))
        local_work_shape = tuple(single_work_dim for _ in range(ndim))

        layers = [
            ParamStripper(('level')),
            AttributeRenamer({'level.solver.problem_operator.apply_op': Name('apply_op', ast.Load())}),
            SemanticFinder(subconfig),
            self.RangeTransformer(shape, ghost_zone, local_work_shape),  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            AttributeGetter({'level': subconfig['level']}),
            ArrayRefIndexTransformer(
                encode_map={'index': 'encode'},
                ndim=ndim),
            PyBasicConversions(),
        ]
        kernel = apply_all_layers(layers, kernel)

        params = kernel.params

        defn = []

        beta_def = ArrayDef(
            SymbolRef("beta_face_values", sym_type=ctypes.POINTER(ctypes.c_double)()),
            size=ndim,
            body=Array(
                body=[
                    SymbolRef("beta_face_values_{}".format(i)) for i in range(ndim)
                ]
            )
        )


        defn.append(beta_def)
        defn.extend(kernel.defn)
        kernel.defn = defn


        kernel.set_kernel()
        kernel.name = "apply_op_kernel"

        for call in kernel.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()  #remove level

        beta_face_values = kernel.find(SymbolRef, name='beta_face_values')
        beta_face_values.set_global()

        macro_func = to_macro_function(subconfig['level'].solver.problem_operator.apply_op,
                                       rename={"level.beta_face_values": SymbolRef("beta_face_values"),
                                               "level.alpha": SymbolRef("alpha")})

        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        encode_func = encode_func.body[-1]

        params.extend([
            SymbolRef("beta_face_values_{}".format(i), sym_type=ctypes.POINTER(ctypes.c_double)())
            for i in range(ndim)
        ])
        params.append(
            SymbolRef("alpha", sym_type=ctypes.POINTER(ctypes.c_double)())
        )

        for param in params:
            param.type = ctypes.POINTER(ctypes.c_double)()
            param.set_global()

        cfile = CFile(body=[encode_func, macro_func, kernel])
        cfile = include_mover(cfile)
        ocl_file = OclFileWrapper("apply_op_kernel").visit(cfile)
        ocl_file = DeclarationFiller().visit(ocl_file)

        control = new_generate_control("apply_op_control", global_size, local_size, params, [kernel])
        # print(control)
        # raise TypeError
        return [control, ocl_file]
        # return [ocl_file]

    def finalize(self, transform_result, program_config):
        subconfig, tuner = program_config
        device = cl.clGetDeviceIDs()[-1]
        level = subconfig['level']
        local_size = compute_local_work_size(device, level.interior_space)
        global_size = reduce(operator.mul, level.interior_space, 1)

        project = Project(transform_result)
        kernel = project.find(OclFile)
        control = project.find(CFile)

        param_types = [cl.cl_mem for _ in ('target_mesh', 'source_mesh')]
        # beta face values
        param_types.extend([cl.cl_mem]*subconfig['level'].solver.problem_operator.dimensions)
        # alpha
        param_types.append(param_types[-1])

        entry_type = [ctypes.c_int32, cl.cl_command_queue, cl.cl_kernel]
        entry_type.extend(param_types)
        entry_type = ctypes.CFUNCTYPE(*entry_type)

        program = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()
        kernel = program["apply_op_kernel"]
        # kernel.argtypes = tuple(cl.cl_mem for _ in range(len(param_types)))
        kernel.argtypes = param_types

        kernel = KernelRunManager(kernel, global_size, local_size)

        fn = OclApplyOpFunction()
        return fn.finalize("apply_op_control", project, entry_type, level, [kernel])