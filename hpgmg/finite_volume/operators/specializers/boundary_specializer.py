import ast
import ctypes
from ctree.c.nodes import MultiNode, Assign, SymbolRef, Constant, For, Lt, PostInc, FunctionDecl, CFile, Pragma
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import dump, get_ast
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import AttributeRenamer, AttributeGetter, \
    IndexTransformer, IndexOpTransformer, IndexDirectTransformer, ParamStripper

import numpy as np
__author__ = 'nzhang-dev'

class BoundaryCFunction(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, mesh):

        #print(self.entry_point_name, [i.shape for i in flattened])
        self._c_function(mesh.ravel())

class CBoundarySpecializer(LazySpecializedFunction):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['index_{}'.format(i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index, sym_type=ctypes.c_uint64()), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            bottom.body = node.body
            self.generic_visit(bottom)
            return top

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
                IndexOpTransformer(ndim=ndim),
                IndexDirectTransformer(ndim=ndim),
                self.RangeTransformer(),
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
        name = self.original_tree.body[0].name
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
        cfile.backend = 'omp'
        cfile.body.append(
            CppInclude("omp.h")
        )
        cfile = include_mover(cfile)
        return [cfile]
