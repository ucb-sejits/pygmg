import ast
import ctypes
from ctree.c.nodes import MultiNode, For, Assign, SymbolRef, Constant, Lt, PostInc, FunctionCall, FunctionDecl, CFile
from ctree.cpp.nodes import CppInclude
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, to_macro_function, sympy_to_c, \
    include_mover
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder

from ctree.frontend import dump, get_ast
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeGetter, \
    IndexOpTransformer, AttributeRenamer, CallReplacer, IndexDirectTransformer, IndexTransformer

import numpy as np

__author__ = 'nzhang-dev'


class InitializeCFunction(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, mesh, exp, coord_transform):

        return self._c_function(mesh.ravel())


class CInitializeMesh(LazySpecializedFunction):

    class InitializeMeshSubconfig(dict):
        def __hash__(self):
            to_hash = [
                self['level'].space,
                self['level'].ghost_zone,
                self['mesh'].shape,
                str(self['exp']),
                self['coord_transform'].__name__
            ]

    def args_to_subconfig(self, args):
        return self.InitializeMeshSubconfig({
            'self': args[0],
            'level': args[1],
            'mesh': args[2],
            'exp': args[3],
            'coord_transform': args[4]
        })

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['coord_{}'.format(i) for i in range(ndim)]
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
        func_node = tree.body[0]
        func_node.body.pop(0)
        ndim = subconfig['self'].dimensions

        coord_tree = get_ast(subconfig['coord_transform']).body[0].body[-1].value



        coord_layers = [
            SemanticFinder(),
            AttributeGetter({'self': subconfig['level']}),
            IndexOpTransformer(ndim=ndim, encode_func_names={'coord': ''}),
            PyBasicConversions(),
        ]

        coord_tree = apply_all_layers(coord_layers, coord_tree)
        coord_transform = coord_tree.args
        expr = sympy_to_c(subconfig['exp'], 'coord_')

        expr_layers = [
            AttributeRenamer({
                'coord_{}'.format(i) : coord_transform[i] for i in range(ndim)
            })
        ]

        expr = apply_all_layers(expr_layers, expr)

        layers = [
            ParamStripper(('self', 'level', 'exp', 'coord_transform')),
            SemanticFinder(subconfig),
            CallReplacer({
                'func': expr
            }),
            self.RangeTransformer(),
            IndexTransformer(('coord',)),
            IndexDirectTransformer(ndim=ndim, encode_func_names={'coord': 'encode'}),
            PyBasicConversions(),

        ]

        tree = apply_all_layers(layers, tree)
        func_node = tree.find(FunctionDecl)
        func_node.params[0].type = ctypes.POINTER(ctypes.c_double)()

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)

        cfile = CFile(body=[
            tree,
            CppInclude("math.h"),
            encode_func
        ])

        cfile = include_mover(cfile)

        return [cfile]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        fn = InitializeCFunction()
        return fn.finalize(
            transform_result[0].find(FunctionDecl).name,
            Project(transform_result),
            ctypes.CFUNCTYPE(None, np.ctypeslib.ndpointer(
                subconfig['mesh'].dtype,
                1,
                subconfig['mesh'].size
            ))
        )


