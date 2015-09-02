from __future__ import division, print_function
import ast
import atexit
import ctypes
import inspect
import math
from ast import Name
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
import pycl as cl
import operator

from ctree.c.nodes import SymbolRef, CFile, FunctionCall, ArrayDef, Array, For, String, Assign, Constant, Lt, PostInc, \
    Return, NotEq, If, FunctionDecl, Ref, Mul, ArrayRef, Sub, Add, Mul, MultiNode, AddAssign, Div, MulAssign, Mod
from ctree.cpp.nodes import CppInclude
from ctree.tune import MinimizeTime
import numpy as np
from ctree.jit import LazySpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
import time
import hpgmg
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, \
    PyGMGOclConcreteSpecializedFunction, KernelRunManager

from hpgmg.finite_volume.operators.specializers.util import to_macro_function, apply_all_layers, include_mover, \
    LayerPrinter, compute_local_work_size, flattened_to_multi_index, new_generate_control, \
    compute_largest_local_work_size, time_this
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.semantic_transformers.ompsemantics import OmpRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer, FunctionCallTimer
from hpgmg.finite_volume.operators.tune.tuners import SmoothTuningDriver


__author__ = 'nzhang-dev'


class SmoothCFunction(PyGMGConcreteSpecializedFunction):

    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     #print("SmoothCFunction Finalize", entry_point_name)
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        thing, level, working_source, working_target, rhs_mesh, lambda_mesh = args
        c_args = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ] + level.beta_face_values + [level.alpha]
        return c_args, {}

    # def __call__(self, thing, level, working_source, working_target, rhs_mesh, lambda_mesh):
    #     args = [
    #         working_source, working_target,
    #         rhs_mesh, lambda_mesh
    #     ] + level.beta_face_values + [level.alpha]
    #     # args.extend(level.beta_face_values)
    #     # args.append(level.alpha)
    #     #flattened = [arg.ravel() for arg in args]
    #     #self._c_function(*flattened)
    #     self._c_function(*args)


class SmoothOclFunction(PyGMGOclConcreteSpecializedFunction):

    def get_all_args(self, args, kwargs):
        thing, level, working_source, working_target, rhs_mesh, lambda_mesh = args
        args_to_bufferize = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ] + level.beta_face_values + [level.alpha]

        for m in range(len(args_to_bufferize)):
            mesh = args_to_bufferize[m]
            if isinstance(mesh, np.ndarray) and not isinstance(mesh, Mesh):
                mesh = Mesh(mesh.shape)
                mesh.fill(0)
                args_to_bufferize[m] = mesh
        return args_to_bufferize

    def set_dirty_buffers(self, args):
        args[3].buffer.dirty = True

class ResidualOclFunction(PyGMGOclConcreteSpecializedFunction):

    def get_all_args(self, args, kwargs):
        thing, level, working_source, working_target, rhs_mesh, lambda_mesh = args
        args_to_bufferize = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ] + level.beta_face_values + [level.alpha]

        for m in range(len(args_to_bufferize)):
            mesh = args_to_bufferize[m]
            if isinstance(mesh, np.ndarray) and not isinstance(mesh, Mesh):
                mesh = Mesh(mesh.shape)
                mesh.fill(0)
                args_to_bufferize[m] = mesh
        return args_to_bufferize

    def set_dirty_buffers(self, args):
        args[3].buffer.dirty = True


class CSmoothSpecializer(LazySpecializedFunction):

    #argspec = ['source', 'target', 'rhs_mesh', 'lambda_mesh', '*level.beta_face_values', 'level.alpha']

    class SmoothSubconfig(dict):
        def __hash__(self):
            operator = self['self'].operator
            hashed = [
                operator.a, operator.b, operator.h2inv,
                operator.dimensions, operator.is_variable_coefficient,
                operator.ghost_zone, tuple(operator.neighborhood_offsets)
            ]
            for i in ('source', 'target', 'rhs_mesh', 'lambda_mesh'):
                hashed.append(self[i].shape)
            #print(hashed)
            return hash(tuple(hashed))

    def get_tuning_driver(self):
        return SmoothTuningDriver(objective=MinimizeTime())

    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return self.SmoothSubconfig({
            param: arg for param, arg in zip(params, args)
        })

    RangeTransformer = CRangeTransformer

    #@time_this
    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        #print(tuner, end="\t")
        ndim = subconfig['self'].operator.solver.dimensions
        ghost = subconfig['self'].operator.ghost_zone
        subconfig['ghost'] = ghost
        shape = subconfig['level'].interior_space
        layers = [
            ParamStripper(('self', 'level')),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            SemanticFinder(subconfig),
            self.RangeTransformer(cache_hierarchy=tuner),
            #RowMajorInteriorPoints(subconfig),
            AttributeGetter({'self': subconfig['self']}),
            ArrayRefIndexTransformer(
                encode_map={
                    'index': 'encode'
                },
                ndim=ndim
            ),
            PyBasicConversions(),
            #FunctionCallTimer((self.original_tree.body[0].name,)),
            #LayerPrinter(),
        ]
        func = apply_all_layers(layers, func)
        macro_func = to_macro_function(subconfig['self'].operator.apply_op,
                                       rename={"level.beta_face_values": SymbolRef("beta_face_values"),
                                               "level.alpha": SymbolRef("alpha")})
        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]
        #print(macro_func)
        #raise Exception()
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        defn = func.defn
        func.defn = [
            SymbolRef('a_x', sym_type=ctypes.c_double()),
            SymbolRef('b', sym_type=ctypes.c_double())
        ]
        # if subconfig['self'].operator.is_variable_coefficient:
            # needs beta values
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_def = ArrayDef(
            SymbolRef("beta_face_values", sym_type=ctypes.POINTER(ctypes.c_double)()),
            size=ndim,
            body=Array(
                body=[
                    SymbolRef("beta_face_values_{}".format(i)) for i in range(ndim)
                ]
            )
        )
        func.defn.append(beta_def)
        func.params.extend([
            SymbolRef("beta_face_values_{}".format(i), sym_type=ctypes.POINTER(ctypes.c_double)())
            for i in range(ndim)
        ])
        #if subconfig['self'].operator.solver.is_helmholtz:
        func.params.append(
            SymbolRef("alpha", sym_type=ctypes.POINTER(ctypes.c_double)())
        )
        func.defn.extend(defn)
        for call in func.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()  #remove level
        for param in func.params:
            param.type = ctypes.POINTER(ctypes.c_double)()
            param.set_restrict()
        # print(func)
        # print(macro_func)
        # print(encode_func)
        #print(func)
        cfile = CFile(body=[
            func, macro_func, encode_func
        ])
        cfile = include_mover(cfile)
        #print("codegen")
        # print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = SmoothCFunction()
        subconfig = program_config[0]
        param_types = [np.ctypeslib.ndpointer(thing.dtype, len(thing.shape), thing.shape) for thing in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_type = np.ctypeslib.ndpointer(
            beta_sample.dtype,
            len(beta_sample.shape),
            beta_sample.shape
        )
        #if subconfig['self'].operator.is_variable_coefficient:
        param_types.extend(
            [beta_type]*subconfig['self'].operator.dimensions
        )
        #if subconfig['self'].operator.solver.is_helmholtz:
        param_types.append(
            param_types[-1]
        )  # add 1 more for alpha
        #print(dump(self.original_tree))
        name = self.tree.body[0].name
        return fn.finalize(name, Project(transform_result),
                           ctypes.CFUNCTYPE(None, *param_types))

    def __call__(self, *args, **kwargs):
        if hpgmg.finite_volume.CONFIG.tune:
            tune_count = 0
            tune_time = time.time()
            while not self._tuner.is_exhausted():
                #super(CSmoothSpecializer, self).__call__(*args, **kwargs)  # for the cache/codegen
                t = time.time()
                super(CSmoothSpecializer, self).__call__(*args, **kwargs)
                total_time = time.time() - t
                #cprint(total_time)
                self.report(time=total_time)
                tune_count += 1
            tune_time = time.time() - tune_time
            if tune_count:
                def print_report():
                    subconfig = self.args_to_subconfig(args)['level'].interior_space
                    print(subconfig)
                    print("Function:", type(self).__name__, "tuning time:", tune_time,
                          "tune count:", tune_count,
                          "best config:",
                          self._tuner.best_configs[subconfig])
                atexit.register(print_report)
        res = super(CSmoothSpecializer, self).__call__(*args, **kwargs)
        return res



class OmpSmoothSpecializer(CSmoothSpecializer):

    RangeTransformer = OmpRangeTransformer

    def transform(self, tree, program_config):
        stuff = super(OmpSmoothSpecializer, self).transform(tree, program_config)
        stuff[0].config_target = 'omp'
        stuff[0].body.insert(0, CppInclude("omp.h"))
    #     for_loop = stuff[0].find(For)
    #     subconfig = program_config[0]
    #     ndim = subconfig['self'].operator.solver.dimensions
    #     for_loop.pragma = 'omp parallel for private(a_x, b)'.format(ndim)
    #     # last_loop = list(stuff[0].find_all(For))[-1]
    #     # last_loop.body.append(
    #     #     FunctionCall(
    #     #         SymbolRef("printf"),
    #     #         args=[String(r"Threads: %d\n"), FunctionCall(SymbolRef("omp_get_num_threads"))]
    #     #     )
    #     # )
        return stuff


class OclSmoothSpecializer(LazySpecializedFunction):

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

    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return CSmoothSpecializer.SmoothSubconfig({
                param: arg for param, arg in zip(params, args)
            })

    def transform(self, tree, program_config):
        kernel = tree.body[0]

        subconfig, tuner = program_config

        ndim = subconfig['self'].operator.solver.dimensions
        shape = subconfig['level'].interior_space
        ghost_zone = subconfig['self'].operator.ghost_zone
        entire_shape = tuple(dim + g*2 for dim, g in zip(shape, ghost_zone))

        device = cl.clGetDeviceIDs()[-1]
        local_size = compute_local_work_size(device, shape)
        single_work_dim = int(round(local_size ** (1/float(ndim))))
        local_work_shape = tuple(single_work_dim for _ in range(ndim))

        # transform to get the kernel function

        layers = [
            ParamStripper(('self', 'level')),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            SemanticFinder(subconfig),
            self.RangeTransformer(shape, ghost_zone, local_work_shape),
            AttributeGetter({'self': subconfig['self']}),
            ArrayRefIndexTransformer(
                encode_map={
                    'index': 'encode'
                },
                ndim=ndim
            ),
            PyBasicConversions(),
        ]

        kernel = apply_all_layers(layers, kernel)

        params = kernel.params

        defn = [
            SymbolRef('a_x', sym_type=ctypes.c_double()),
            SymbolRef('b', sym_type=ctypes.c_double())
        ]

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
        kernel.name = "smooth_points_kernel"

        for call in kernel.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()  #remove level

        beta_face_values = kernel.find(SymbolRef, name='beta_face_values')
        beta_face_values.set_global()

        # need this because DeclarationFiller does not visit Ocl files

        symbols_to_declare_double = [
            kernel.find(SymbolRef, name='____temp__a_x'),
            kernel.find(SymbolRef, name='____temp__b')
        ]

        for symbol in symbols_to_declare_double:
            if symbol:
                symbol.type = ctypes.c_double()

        # macros and defines

        double_enabler = StringTemplate("""
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            """)

        macro_func = to_macro_function(subconfig['self'].operator.apply_op,
                                       rename={"level.beta_face_values": SymbolRef("beta_face_values"),
                                               "level.alpha": SymbolRef("alpha")})
        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        encode_func = encode_func.body[-1]

        ocl_include = StringTemplate("""
            #include <stdio.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)

        # parameter management

        params.extend([
            SymbolRef("beta_face_values_{}".format(i), sym_type=ctypes.POINTER(ctypes.c_double)())
            for i in range(ndim)
        ])
        params.append(
            SymbolRef("alpha", sym_type=ctypes.POINTER(ctypes.c_double)())
        )

        for param in params:
            param.type = ctypes.POINTER(ctypes.c_double)()
            param.set_global()  # this is not safe
            # if param.name != 'working_target':
            #     param.set_const()

        # file creation

        ocl_file = OclFile(name="smooth_points_kernel",
                           body=[
                               double_enabler,
                               encode_func,
                               macro_func,
                               kernel,
                           ])

        global_size = reduce(operator.mul, shape, 1)
        # control = new_generate_control("smooth_points_control", global_size, local_size, params, [ocl_file])
        # return [control, ocl_file]
        return [ocl_file]

    def finalize(self, transform_result, program_config):

        subconfig, tuner = program_config
        device = cl.clGetDeviceIDs()[-1]
        level = subconfig['level']
        local_size = compute_local_work_size(device, level.interior_space)
        global_size = reduce(operator.mul, level.interior_space, 1)

        project = Project(transform_result)
        kernel = transform_result[0]
        # control = project.find(CFile)

        param_types = [cl.cl_mem for _ in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        # beta face values
        param_types.extend([cl.cl_mem]*subconfig['self'].operator.dimensions)
        # alpha
        param_types.append(param_types[-1])


        # entry_type = [ctypes.c_int32, cl.cl_command_queue, cl.cl_kernel]
        # entry_type.extend(param_types)
        # entry_type = ctypes.CFUNCTYPE(*entry_type)

        program = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()
        kernel = program["smooth_points_kernel"]
        kernel.argtypes = param_types

        kernel = KernelRunManager(kernel, global_size, local_size)

        fn = SmoothOclFunction()
        # return fn.finalize("smooth_points_control", project, entry_type, level, [kernel])
        return fn.finalize(project, level, [kernel])

class OclResidualSpecializer(LazySpecializedFunction):

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

    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return CSmoothSpecializer.SmoothSubconfig({
                param: arg for param, arg in zip(params, args)
            })

    def transform(self, tree, program_config):
        kernel = tree.body[0]

        subconfig, tuner = program_config

        ndim = subconfig['self'].operator.solver.dimensions
        shape = subconfig['level'].interior_space
        ghost_zone = subconfig['self'].operator.ghost_zone
        entire_shape = tuple(dim + g*2 for dim, g in zip(shape, ghost_zone))

        device = cl.clGetDeviceIDs()[-1]
        local_size = compute_local_work_size(device, shape)
        single_work_dim = int(round(local_size ** (1/float(ndim))))
        local_work_shape = tuple(single_work_dim for _ in range(ndim))

        # transform to get the kernel function

        layers = [
            ParamStripper(('self', 'level')),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            SemanticFinder(subconfig),
            self.RangeTransformer(shape, ghost_zone, local_work_shape),
            AttributeGetter({'self': subconfig['self']}),
            ArrayRefIndexTransformer(
                encode_map={
                    'index': 'encode'
                },
                ndim=ndim
            ),
            PyBasicConversions(),
        ]

        kernel = apply_all_layers(layers, kernel)

        params = kernel.params

        defn = [
            SymbolRef('a_x', sym_type=ctypes.c_double()),
            SymbolRef('b', sym_type=ctypes.c_double())
        ]

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
        kernel.name = "smooth_points_kernel"

        for call in kernel.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()  #remove level

        beta_face_values = kernel.find(SymbolRef, name='beta_face_values')
        beta_face_values.set_global()

        # need this because DeclarationFiller does not visit Ocl files

        symbols_to_declare_double = [
            kernel.find(SymbolRef, name='____temp__a_x'),
            kernel.find(SymbolRef, name='____temp__b')
        ]

        for symbol in symbols_to_declare_double:
            if symbol:
                symbol.type = ctypes.c_double()

        # macros and defines

        double_enabler = StringTemplate("""
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            """)

        macro_func = to_macro_function(subconfig['self'].operator.apply_op,
                                       rename={"level.beta_face_values": SymbolRef("beta_face_values"),
                                               "level.alpha": SymbolRef("alpha")})
        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        encode_func = encode_func.body[-1]

        ocl_include = StringTemplate("""
            #include <stdio.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)

        # parameter management

        params.extend([
            SymbolRef("beta_face_values_{}".format(i), sym_type=ctypes.POINTER(ctypes.c_double)())
            for i in range(ndim)
        ])
        params.append(
            SymbolRef("alpha", sym_type=ctypes.POINTER(ctypes.c_double)())
        )

        for param in params:
            param.type = ctypes.POINTER(ctypes.c_double)()
            param.set_global()  # this is not safe
            # if param.name != 'working_target':
            #     param.set_const()

        # file creation

        ocl_file = OclFile(name="smooth_points_kernel",
                           body=[
                               double_enabler,
                               encode_func,
                               macro_func,
                               kernel,
                           ])

        global_size = reduce(operator.mul, shape, 1)
        # control = new_generate_control("smooth_points_control", global_size, local_size, params, [ocl_file])
        # return [control, ocl_file]
        return [ocl_file]

    def finalize(self, transform_result, program_config):

        subconfig, tuner = program_config
        device = cl.clGetDeviceIDs()[-1]
        level = subconfig['level']
        local_size = compute_local_work_size(device, level.interior_space)
        global_size = reduce(operator.mul, level.interior_space, 1)

        project = Project(transform_result)
        kernel = transform_result[0]
        # control = project.find(CFile)

        param_types = [cl.cl_mem for _ in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        # beta face values
        param_types.extend([cl.cl_mem]*subconfig['self'].operator.dimensions)
        # alpha
        param_types.append(param_types[-1])


        # entry_type = [ctypes.c_int32, cl.cl_command_queue, cl.cl_kernel]
        # entry_type.extend(param_types)
        # entry_type = ctypes.CFUNCTYPE(*entry_type)

        program = cl.clCreateProgramWithSource(level.context, kernel.codegen()).build()
        kernel = program["smooth_points_kernel"]
        kernel.argtypes = param_types

        kernel = KernelRunManager(kernel, global_size, local_size)

        fn = ResidualOclFunction()
        # return fn.finalize("smooth_points_control", project, entry_type, level, [kernel])
        return fn.finalize(project, level, [kernel])
