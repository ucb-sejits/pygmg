from __future__ import division, print_function
import ast
import ctypes
import inspect
import math
from ast import Name
import operator
from ctree.c.macros import NULL
from ctree.ocl import get_context_and_queue_from_devices
from ctree.templates.nodes import StringTemplate
from ctree.transforms.declaration_filler import DeclarationFiller
import pycl as cl

from ctree.c.nodes import SymbolRef, CFile, FunctionCall, ArrayDef, Array, For, String, Assign, Constant, Lt, PostInc, \
    Block, MultiNode, Add, Mul, ArrayRef, Div, Sub, FunctionDecl, Return, If, NotEq, Ref, Eq, AugAssign, AddAssign
from ctree.cpp.nodes import CppInclude, CppDefine
from ctree.ocl.nodes import OclFile
import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
import time

from hpgmg.finite_volume.operators.specializers.util import to_macro_function, apply_all_layers, include_mover, \
    LayerPrinter, compute_local_work_size
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer, FunctionCallTimer


__author__ = 'nzhang-dev'


class SmoothCFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, working_source, working_target, rhs_mesh, lambda_mesh):
        args = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ]
        #if thing.operator.is_variable_coefficient:
        args.extend(level.beta_face_values)
          #  if thing.operator.solver.is_helmholtz:
        args.append(level.alpha)
        flattened = [arg.ravel() for arg in args]
        #print(self.entry_point_name, [i.shape for i in flattened])
        #t = time.time()

        # print("new specializer")
        # print("")

        # for mesh in flattened:
            # print(np.linalg.norm(mesh), )
            # print(mesh)


        self._c_function(*flattened)
        #print("C-Call: {}".format(time.time() - t), end="\t")


class SmoothOclFunction(ConcreteSpecializedFunction):

    def __init__(self):
        device = cl.clGetDeviceIDs()[-1]
        self.context, self.queue = get_context_and_queue_from_devices([device])
        # self.max_work_group_size = device.max_work_group_size
        self.kernel = []
        self._c_function = lambda: 0

    def finalize(self, entry_point_name, project_node, entry_point_typesig, kernel):
        self.kernel = kernel
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, working_source, working_target, rhs_mesh, lambda_mesh):
        args = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ]
        args.extend(level.beta_face_values)
        args.append(level.alpha)
        flattened = [arg.ravel() for arg in args]

        buffers_and_events = [cl.buffer_from_ndarray(self.queue, ary=mesh) for mesh in flattened]

        buffers = [b_e[0] for b_e in buffers_and_events]

        arguments = [self.queue, self.kernel]

        arguments.extend(buffers)

        # print("new specializer")
        # print("")

        # for mesh in flattened:
            # print(np.linalg.norm(mesh), )
            # print(mesh)


        self._c_function(*arguments)

        for i in range(len(buffers)):
            cl.buffer_to_ndarray(self.queue, buffers[i], out=flattened[i])


class CSmoothSpecializer(LazySpecializedFunction):

    argspec = ['source', 'target', 'rhs_mesh', 'lambda_mesh', '*level.beta_face_values', 'level.alpha']

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['index_{}'.format(i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            bottom.body = node.body
            self.generic_visit(bottom)
            return top

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


    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return self.SmoothSubconfig({
            param: arg for param, arg in zip(params, args)
        })

    #@time_this
    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['self'].operator.solver.dimensions
        ghost = subconfig['self'].operator.ghost_zone
        subconfig['ghost'] = ghost
        #shape = subconfig['self'].operator.
        layers = [
            ParamStripper(('self', 'level')),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            SemanticFinder(subconfig),
            self.RangeTransformer(),
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
        # print(func)
        # print(macro_func)
        # print(encode_func)
        #print(func)
        cfile = CFile(body=[
            func, macro_func, encode_func
        ])
        cfile = include_mover(cfile)
        #print("codegen")
        print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = SmoothCFunction()
        subconfig = program_config[0]
        param_types = [np.ctypeslib.ndpointer(thing.dtype, 1, thing.size) for thing in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_type = np.ctypeslib.ndpointer(
                beta_sample.dtype,
                1,
                beta_sample.size
            )
        #if subconfig['self'].operator.is_variable_coefficient:
        param_types.extend(
            [beta_type]*subconfig['self'].operator.dimensions
        )
        #if subconfig['self'].operator.solver.is_helmholtz:
        param_types.append(
            param_types[-1]
        ) # add 1 more for alpha
        #print(dump(self.original_tree))
        name = self.tree.body[0].name
        return fn.finalize(name, Project(transform_result),
                    ctypes.CFUNCTYPE(None, *param_types))

    def __call__(self, *args, **kwargs):
        #t = time.time()
        res = super(CSmoothSpecializer, self).__call__(*args, **kwargs)
        # print("LSF Call: ", time.time() - t)
        return res



class OmpSmoothSpecializer(CSmoothSpecializer):

    class RangeTransformer(ast.NodeTransformer):
        def __init__(self, block_hierarchy):
            self.block_hierarchy = block_hierarchy

        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['index_{}'.format(i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            top.pragma = 'omp parallel for collapse(2)'
            bottom.body = node.body
            self.generic_visit(bottom)
            return top

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


class OclSmoothSpecializer(CSmoothSpecializer):

    class RangeTransformer(ast.NodeTransformer):

        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            thread_id = Assign(SymbolRef("thread_id", ctypes.c_int()),
                                          Add(Mul(FunctionCall(SymbolRef("get_global_id"), [0]),
                                                  FunctionCall(SymbolRef("get_local_size"), [0])),
                                              FunctionCall(SymbolRef("get_local_id"), [0])))
            indices_decl = ArrayRef(SymbolRef("indices", ctypes.c_int()), Constant(ndim))
            decode_call = FunctionCall(SymbolRef("decode"), [SymbolRef("thread_id"), SymbolRef("indices")])
            indices_assign = []
            for d in range(ndim):
                indices_assign.append(Assign(SymbolRef("index_{}".format(d), ctypes.c_int()),
                                             ArrayRef(SymbolRef("indices"), Constant(d))))

            body = []
            body.append(thread_id)
            body.append(indices_decl)
            body.append(decode_call)
            body.extend(indices_assign)
            body.extend(node.body)

            return MultiNode(body)

    def generate_decode(self, dims, ghost_zone):

        body=[]
        ndim = len(dims)

        multipliers = [reduce(operator.mul, dims[i+1:], 1) for i in range(ndim)]
        for d in range(ndim):
            if d == 0:
                dividend = SymbolRef("thread_id")
                previous = Mul(ArrayRef(SymbolRef("indices"), Constant(0)), Constant(multipliers[0]))
            else:
                dividend = Sub(SymbolRef("thread_id"), previous)
                previous = Add(previous, Mul(ArrayRef(SymbolRef("indices"), Constant(d)), Constant(multipliers[d])))
            index = Div(dividend, Constant(multipliers[d]))
            assign = Assign(ArrayRef(SymbolRef("indices"), Constant(d)), index)
            body.append(assign)

        for i in range(ndim):
            body.append(AddAssign(ArrayRef(SymbolRef("indices"), Constant(i)), Constant(ghost_zone[i])))

        return FunctionDecl(name="decode",
                            params=[SymbolRef("thread_id", ctypes.c_int()),
                                    SymbolRef("indices", ctypes.POINTER(ctypes.c_int)())],
                            defn=body)

    def generate_control(self, global_work_dims, kernel):

        ndim = len(global_work_dims)

        params = [SymbolRef("queue", cl.cl_command_queue()), SymbolRef("kernel", cl.cl_kernel())]
        for param in range(len(kernel.params)):
            params.append(SymbolRef("buf_{}".format(param), cl.cl_mem()))

        defn = []

        defn.append(ArrayDef(SymbolRef("global", ctypes.c_ulong()), ndim, Array(body=[Constant(reduce(operator.mul, global_work_dims, 1))])))

        device = cl.clGetDeviceIDs()[-1]

        local_work_size = compute_local_work_size(device, global_work_dims)

        defn.append(ArrayDef(SymbolRef("local", ctypes.c_ulong()), ndim, Array(body=[Constant(local_work_size)])))  # write local size computer??

        defn.append(Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0)))

        enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
            SymbolRef("queue"), SymbolRef("kernel"),
            Constant(1), NULL(),
            SymbolRef("global"), SymbolRef("local"),
            Constant(0), NULL(), NULL()
        ])

        for param in range(len(kernel.params)):
            defn.append(FunctionCall(SymbolRef("clSetKernelArg"),
                                     [SymbolRef("kernel"),
                                      Constant(param),
                                      FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
                                      Ref(SymbolRef("buf_{}".format(param)))]))

        defn.extend(check_ocl_error(enqueue_call, "clEnqueueNDRangeKernel"))

        finish_call = check_ocl_error(
            FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')]),
            "clFinish"
        )
        defn.extend(finish_call)
        defn.append(Return(SymbolRef("error_code")))

        return FunctionDecl(ctypes.c_int(), "smooth_points_control", params, defn)

    def transform(self, tree, program_config):

        subconfig, tuner = program_config
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        ndim = subconfig['self'].operator.solver.dimensions
        dims = [int(round(2**(bits_per_dim - 1)))] * ndim

        ghost = subconfig['self'].operator.ghost_zone
        subconfig['ghost'] = ghost

        interior_space = tuple(point for point in subconfig['level'].interior_space)

        decode_func = self.generate_decode(interior_space, ghost)

        stuff = super(OclSmoothSpecializer, self).transform(tree, program_config)
        stuff[0] = DeclarationFiller().visit(stuff[0])

        kernel = stuff[0].find(FunctionDecl)
        kernel.set_kernel()
        beta_face_values = kernel.find(SymbolRef, name='beta_face_values')
        beta_face_values.set_global()
        for param in kernel.params:
            param.set_global()
        encode = stuff[0].find(CppDefine, name='encode')
        apply_op = stuff[0].find(CppDefine, name='apply_op')
        ocl_include = StringTemplate("""
            #include <stdio.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
        double_enable = StringTemplate("""
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            """)
        ocl_file = OclFile(kernel.name, body=[double_enable, apply_op, encode, decode_func, kernel])

        control = CFile(name="smooth_points_control", body=[ocl_include, self.generate_control(interior_space, kernel)])

        control.config_target = 'opencl'

        print(ocl_file)
        print(control)

        return [control, ocl_file]

    def finalize(self, transform_result, program_config):

        project = Project(transform_result)
        kernel = project.find(OclFile)
        control = project.find(CFile)
        fn = SmoothOclFunction()
        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        stencil_kernel_ptr = program[kernel.name]
        entry_type = [ctypes.c_int32, cl.cl_command_queue, cl.cl_kernel]
        smooth_points_func = kernel.find(FunctionDecl, name='smooth_points')
        entry_type.extend(cl.cl_mem for _ in range(8))
        # entry_type.extend(cl.cl_mem for _ in range(len(smooth_points_func.params)))

        entry_type = ctypes.CFUNCTYPE(*entry_type)

        return fn.finalize("smooth_points_control", project, entry_type, stencil_kernel_ptr)


        subconfig = program_config[0]
        param_types = [np.ctypeslib.ndpointer(thing.dtype, 1, thing.size) for thing in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_type = np.ctypeslib.ndpointer(
                beta_sample.dtype,
                1,
                beta_sample.size
            )
        #if subconfig['self'].operator.is_variable_coefficient:
        param_types.extend(
            [beta_type]*subconfig['self'].operator.dimensions
        )
        #if subconfig['self'].operator.solver.is_helmholtz:
        param_types.append(
            param_types[-1]
        ) # add 1 more for alpha
        #print(dump(self.original_tree))
        name = self.tree.body[0].name
        return fn.finalize(name, Project(transform_result),
                    ctypes.CFUNCTYPE(None, *param_types))


def check_ocl_error(code_block, message="kernel"):
    return [
        Assign(
            SymbolRef("error_code"),
            code_block
        ),
        If(
            NotEq(SymbolRef("error_code"), SymbolRef("CL_SUCCESS")),
            [
                FunctionCall(
                    SymbolRef("printf"),
                    [
                        String("OPENCL ERROR: {}:error code \
                               %d\\n".format(message)),
                        SymbolRef("error_code")
                    ]
                ),
                Return(SymbolRef("error_code")),
            ]
        )
    ]

