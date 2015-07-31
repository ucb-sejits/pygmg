import ast
import ctypes
from ctree.ocl.nodes import OclFile
import pycl as cl
from ctree.c.nodes import MultiNode, Assign, SymbolRef, Constant, For, Lt, PostInc, FunctionDecl, CFile, Pragma, \
    FunctionCall, String
from ctree.cpp.nodes import CppInclude, CppDefine
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import dump, get_ast
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import AttributeRenamer, AttributeGetter, \
    IndexTransformer, IndexOpTransformer, IndexDirectTransformer, ParamStripper, OclFileWrapper

import numpy as np
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


class BoundaryOclFunction(ConcreteSpecializedFunction): #PyGMGConcreteSpecializedFunction???

    def __init__(self, kernels):
        self.kernels = kernels
        self._c_function = lambda: 0

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, mesh):
        for kernel, k_idx in zip(self.kernels, range(len(self.kernels))):
            kernel.argtypes = (cl.cl_mem,)
            run_evt = kernel(level.buffers[0]).on(level.queue, 1)
            run_evt.wait()
            # level.boundary_events.append(run_evt)


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

    def args_to_subconfig(self, args):
        return CBoundarySpecializer.BoundarySpecializerSubconfig({
            'self': args[0],
            'level': args[1],
            'mesh': args[2]
        })

    def transform(self, tree, program_config):
        subconfig, tuning_config = program_config
        ndim = subconfig['mesh'].ndim
        kernel_files = []

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64).find(CppDefine, name="encode")

        for boundary, kernel, k_idx in zip(subconfig['self'].boundary_cases(),
                                           subconfig['self'].kernels,
                                           range(len(subconfig['self'].kernels))):
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
                ParamStripper(('level',)),
                PyBasicConversions(),
                OclFileWrapper(name="kernel_%d" % k_idx),
            ]
            ocl_file = apply_all_layers(layers, kernel_tree)
            func = ocl_file.find(FunctionDecl, name="kernel")
            func.name = "kernel_%d" % k_idx
            func.set_kernel()
            func.params[0].type = ctypes.POINTER(ctypes.c_double)()
            func.params[0].set_global()

            ocl_file.body.append(encode_func)
            ocl_file = include_mover(ocl_file)
            kernel_files.append(ocl_file)

        # make all the kernels separate files, and enqueue them all in the call

        c_file = CFile(name="dummy_c_file",
                       body=[FunctionDecl(name="dummy_func", defn=[Assign(SymbolRef("x", ctypes.c_int()), Constant(5))],
                                          params=[SymbolRef("z", ctypes.c_int())])])
        c_file.config_target = 'opencl'
        files = [c_file]
        files.extend(kernel_files)
        return files

    def finalize(self, transform_result, program_config):
        subconfig, tuner = program_config
        project = Project(transform_result)
        kernels = [kernel for kernel in project.find_all(OclFile)]
        kernels = [cl.clCreateProgramWithSource(subconfig['level'].context, kernel.codegen()).build() for kernel in kernels]
        kernels = [kernel["kernel_%d" % k_idx] for kernel, k_idx in zip(kernels, range(len(kernels)))]
        return BoundaryOclFunction(kernels).finalize("dummy_func", project, ctypes.CFUNCTYPE(ctypes.c_int))

