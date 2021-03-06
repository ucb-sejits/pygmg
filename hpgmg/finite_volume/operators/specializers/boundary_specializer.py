import ast
import ctypes
from ctree.c.macros import NULL
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
import pycl as cl
from ctree.c.nodes import MultiNode, Assign, SymbolRef, Constant, For, Lt, PostInc, FunctionDecl, CFile, Pragma, \
    FunctionCall, String, ArrayDef, Array, Ref, Return, ArrayRef
from ctree.cpp.nodes import CppInclude, CppDefine
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import dump, get_ast
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, KernelRunManager, \
    PyGMGOclConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover, flattened_to_multi_index, \
    time_this, compute_largest_local_work_size
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.semantic_transformers.oclsemantics import OclRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import AttributeRenamer, AttributeGetter, \
    IndexTransformer, IndexOpTransformer, IndexDirectTransformer, ParamStripper

import numpy as np
import operator

__author__ = 'nzhang-dev'

class BoundaryCFunction(PyGMGConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        return [args[2].ravel()], {}  # mesh.ravel

    def __call__(self, thing, level, mesh):

        #print(self.entry_point_name, [i.shape for i in flattened])
        self._c_function(mesh.ravel())


class BoundaryOclFunction(PyGMGOclConcreteSpecializedFunction):

    # def __call__(self, *args, **kwargs):
    #     args_to_bufferize = self.get_all_args(args, kwargs)
    #
    #     self.set_kernel_args(args_to_bufferize, kwargs)
    #
    #     for kernel in self.kernels:
    #         kernel.kernel(*kernel.args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
    #         # run_evt = kernel.kernel(*kernel_args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
    #         # run_evt.wait()
    #     self.set_dirty_buffers(args)
    #     return self.reduced_value()

    def get_all_args(self, args, kwargs):
        mesh = args[2]
        return [mesh]

    def set_dirty_buffers(self, args):
        args[2].buffer.dirty = True


class CBoundarySpecializer(LazySpecializedFunction):

    class BoundarySpecializerSubconfig(dict):
        def __hash__(self):
            level = self['level']
            hashed = (
                level.space, level.ghost_zone, self['self'].name
            )
            return hash(hashed)

    def args_to_subconfig(self, args):
        return self.BoundarySpecializerSubconfig({
            'self': args[0],
            'level': args[1],
            'mesh': args[2]
        })

    def transform(self, tree, program_config):
        subconfig, tuning_config = program_config
        ndim = subconfig['mesh'].ndim
        kernel_bodies = MultiNode()
        for boundary, kernel in zip(subconfig['self'].boundary_cases(), subconfig['self'].kernels):
            kernel_tree = get_ast(kernel)
            namespace = {'kernel': kernel}
            namespace.update(subconfig)
            layers = [
                AttributeRenamer({
                    'boundary': ast.Tuple(elts=[ast.Num(n=i) for i in boundary], ctx=ast.Load()),
                }),
                SemanticFinder(namespace=subconfig, locals={}),
                AttributeGetter(namespace),
                IndexTransformer(indices=('index',)),
                IndexOpTransformer(ndim=ndim, encode_func_names={'index': 'encode'}),
                IndexDirectTransformer(ndim=ndim),
                CRangeTransformer(),
                PyBasicConversions()
            ]
            kernel_tree = apply_all_layers(layers, kernel_tree)
            #print(dump(kernel_tree))
            kernel_bodies.body.extend(
                kernel_tree.body[0].defn
            )
        c_func = tree.body[0]
        layers = [
            ParamStripper(('self', 'level')),
            PyBasicConversions(),
        ]
        c_func = apply_all_layers(layers, c_func)
        c_func.defn = kernel_bodies.body
        c_func.params[0].type = ctypes.POINTER(ctypes.c_double)()
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        cfile = CFile(body=[c_func, encode_func])
        cfile = include_mover(cfile)
        #return
        return [cfile]


    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        fn = BoundaryCFunction()
        name = self.tree.body[0].name
        mesh = subconfig['mesh']
        return fn.finalize(
            name,
            Project(transform_result),
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(mesh.dtype, 1, mesh.size))
        )

class OmpBoundarySpecializer(CBoundarySpecializer):
    def transform(self, tree, program_config):
        cfile = super(OmpBoundarySpecializer, self).transform(tree, program_config)[0]

        #because of the way we ordered the kernels, we can do simple task grouping
        #every kernel depends only on a subset of the kernels whose norm(boundary) is less than itself.
        def num_kernels(norm, ndim):
            """
            calculates the number of kernels with 1-norm norm given ndim dimensions
            :param norm: 1-norm
            :param ndim: number of dimensions
            :return: number of kernels with 1-norm norm
            """

            return 2**norm * np.math.factorial(ndim) / (np.math.factorial(norm) * np.math.factorial(ndim - norm))

        def chunkify(lst, breakdown):
            i = iter(lst)
            return [
                [next(i) for _ in range(size)]
                for size in breakdown
            ]

        subconfig, tuner_config = program_config
        ndim = subconfig['mesh'].ndim
        kernel_breakdown = [num_kernels(norm, ndim) for norm in range(1, ndim+1)]
        decl = cfile.find(FunctionDecl)
        kernels = decl.defn
        breakdown = chunkify(kernels, kernel_breakdown)
        new_defn = [Pragma(pragma="omp parallel", body=[], braces=True)]
        # new_defn[0].body.append(
        #     FunctionCall(SymbolRef('printf'), args=[String(r"%d\n"),
        #                                             FunctionCall(SymbolRef("omp_get_num_threads"))])
        # )
        for parallelizable in breakdown:
            pragma = Pragma(pragma="omp taskgroup", braces=True, body=[])
            for loop in parallelizable:
                pragma.body.append(
                    Pragma(
                        pragma="omp task",
                        body=[loop],
                        braces=True
                    )
                )
            new_defn[0].body.append(pragma)
        decl.defn = new_defn
        cfile.config_target = 'omp'
        cfile.body.append(
            CppInclude("omp.h")
        )
        cfile = include_mover(cfile)
        return [cfile]


class OclBoundarySpecializer(LazySpecializedFunction):

    class RangeTransformer(ast.NodeTransformer):

        def visit_RangeNode(self, node):
            body=[
                Assign(SymbolRef("global_id"), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
            ]
            ranges = node.iterator.ranges
            offsets = tuple(r[0] for r in ranges)
            shape = tuple(r[1] - r[0] for r in ranges)
            indices = flattened_to_multi_index(SymbolRef("global_id"), shape, offsets=offsets)
            for d in range(len(shape)):
                body.append(Assign(SymbolRef("index_%d"%d), indices[d]))
            body.extend(node.body)
            return MultiNode(body=body)

    def args_to_subconfig(self, args):
        return CBoundarySpecializer.BoundarySpecializerSubconfig({
            'self': args[0],
            'level': args[1],
            'mesh': args[2]
        })

    def transform(self, tree, program_config):
        subconfig, tuning_config = program_config
        ndim = subconfig['mesh'].ndim

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64).find(CppDefine, name="encode")

        ocl_file_bodies = []
        for dim in range(ndim):
            body = []
            defn = []
            defn.append(SymbolRef("global_id", ctypes.c_ulong()))
            defn.extend(SymbolRef("index_%d"%d, ctypes.c_ulong()) for d in range(ndim))
            func = FunctionDecl(name="kernel_%d"%dim, params=[], defn=defn)
            body.append(func)
            ocl_file_bodies.append(body)

        num_kernels_per_dim = [calc_num_k_dim_cubes(ndim, ndim - dim - 1) for dim in range(ndim)]

        boundary_map = {}
        for dim in range(ndim):
            for i in range(num_kernels_per_dim[dim]):
                k_idx = sum(num_kernels_per_dim[:dim]) + i
                boundary_map[k_idx] = dim

        for boundary, kernel, k_idx in zip(subconfig['self'].boundary_cases(),
                                           subconfig['self'].kernels,
                                           range(len(subconfig['self'].kernels))):

            current_dim = boundary_map[k_idx]

            kernel_tree = get_ast(kernel)
            namespace = {'kernel': kernel}
            namespace.update(subconfig)
            layers = [
                AttributeRenamer({
                    'boundary': ast.Tuple(elts=[ast.Num(n=i) for i in boundary], ctx=ast.Load()),
                }),
                SemanticFinder(namespace=subconfig, locals={}),
                AttributeGetter(namespace),
                IndexTransformer(indices=('index',)),
                IndexOpTransformer(ndim=ndim, encode_func_names={'index': 'encode'}),
                IndexDirectTransformer(ndim=ndim),
                self.RangeTransformer(),
                # OclRangeTransformer(),
                ParamStripper(('level',)),
                PyBasicConversions(),
            ]
            transformed_file = apply_all_layers(layers, kernel_tree)
            func = transformed_file.find(FunctionDecl, name="kernel")
            if not ocl_file_bodies[current_dim][0].params:
               ocl_file_bodies[current_dim][0].params = func.params
            ocl_file_bodies[current_dim][0].defn.extend(func.defn)

        double_include = StringTemplate("""#pragma OPENCL EXTENSION cl_khr_fp64: enable""")

        for body in ocl_file_bodies:
            func = body[0]
            func.set_kernel()
            func.params[0].type = ctypes.POINTER(ctypes.c_double)()
            func.params[0].set_global()

        for body in ocl_file_bodies:
            body.insert(0, double_include)
            body.append(encode_func)

        ocl_files = [OclFile(name="kernel_%d"%d, body=ocl_file_bodies[d]) for d in range(ndim)]
        ocl_files = [include_mover(file) for file in ocl_files]

        ocl_include = StringTemplate("""
            #include <stdio.h>
            #include <time.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)

        c_file = CFile(name="boundary_control", body=[ocl_include, generate_control(subconfig['level'].interior_space)])
        c_file.config_target = 'opencl'
        files = [c_file]
        # print(c_file)
        # raise TypeError
        files.extend(ocl_files)
        return files
        # return ocl_files

    def finalize(self, transform_result, program_config):
        subconfig, tuner = program_config
        level = subconfig['level']
        interior_space = level.interior_space
        ndim = len(interior_space)
        project = Project(transform_result)
        control = project.find(CFile)
        kernels = project.files[1:]

        global_sizes = [reduce(operator.mul, interior_space[dim+1:], 1) for dim in range(ndim)]
        local_sizes = [compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], gsize) for gsize in global_sizes]

        entry_type = [ctypes.c_int32, cl.cl_command_queue]
        entry_type.extend(cl.cl_kernel for _ in range(ndim))
        entry_type.append(cl.cl_mem)
        entry_type.append(np.ctypeslib.ndpointer(np.float32, 1, (1,)))
        entry_type = ctypes.CFUNCTYPE(*entry_type)

        kernels = [cl.clCreateProgramWithSource(level.context, kernel.codegen()).build() for kernel in kernels]
        kernels = [kernel["kernel_%d" % k_idx] for kernel, k_idx in zip(kernels, range(len(kernels)))]
        for dim in range(ndim):
            kernels[dim].argtypes = (cl.cl_mem,)
            kernels[dim] = KernelRunManager(kernels[dim], global_sizes[dim], local_sizes[dim])

        fn = BoundaryOclFunction()
        fn = fn.finalize("boundary_control", project, entry_type, level, kernels)
        # fn = fn.finalize(project, level, kernels)
        return fn


def calc_num_k_dim_cubes(ndim, k):
    return ((2 ** (ndim - k)) * choose(ndim, k))

def choose(n, k):
    from math import factorial
    return factorial(n) / factorial(k) / factorial(n - k)

def generate_control(interior_space, params=None):
    # generates c function that enqueues all boundary kernels
    # params will be the queue, 3 kernels, and 1 buffer
    # need to calculate global and local sizes for each and enqueue

    ndim = len(interior_space)

    #parameters
    control_params = []
    control_params.append(SymbolRef("queue", cl.cl_command_queue()))
    control_params.extend(SymbolRef("kernel_%d"%k_idx, cl.cl_kernel()) for k_idx in range(ndim))
    control_params.append(SymbolRef("working_source", cl.cl_mem()))
    control_params.append(SymbolRef("total_time", ctypes.POINTER(ctypes.c_float)()))

    #definition
    defn = [
        ArrayRef(SymbolRef("global", ctypes.c_ulong()), Constant(1)),
        ArrayRef(SymbolRef("local", ctypes.c_ulong()), Constant(1))
    ]

    defn.append(StringTemplate("""clock_t start, diff;"""))

    for k_idx in range(ndim):
        set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                               [
                                   SymbolRef("kernel_%d"%k_idx),
                                   Constant(0),
                                   FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
                                   Ref(SymbolRef("working_source"))
                               ])
        defn.append(set_arg)
    defn.append(StringTemplate("""start = clock();"""))
    for k_idx in range(ndim):
        global_size = reduce(operator.mul, interior_space[k_idx + 1:], 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        defn.append(Assign((ArrayRef(SymbolRef("global"), Constant(0))), Constant(global_size)))
        defn.append(Assign((ArrayRef(SymbolRef("local"), Constant(0))), Constant(local_size)))
        enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
            SymbolRef("queue"),
            SymbolRef("kernel_%d"%k_idx),
            Constant(1),
            NULL(),
            SymbolRef("global"),
            SymbolRef("local"),
            Constant(0),
            NULL(),
            NULL()
        ])
        defn.append(enqueue_call)
    defn.append(StringTemplate("""clFinish(queue);"""))
    defn.append(StringTemplate("""total_time[0] = total_time[0] + (diff / CLOCKS_PER_SEC);"""))

    defn.append(Return(Constant(0)))
    func = FunctionDecl(return_type=ctypes.c_int(), name="boundary_control", params=control_params, defn=defn)
    return func

