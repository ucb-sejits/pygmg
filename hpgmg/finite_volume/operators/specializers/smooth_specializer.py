import ast
from ctree.c.nodes import SymbolRef
from ctree.jit import LazySpecializedFunction
from ctree.transformations import PyBasicConversions
from hpgmg.finite_volume.operators.specializers.util import to_macro_function, apply_all_layers, LayerPrinter
from ctree.frontend import dump, get_ast
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter

from ast import Name

__author__ = 'nzhang-dev'


def jit_smooth(func):

    specialized = SmoothSpecializer(get_ast(func))
    def wrapper(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        to_macro_function(self.operator.apply_op)
        return specialized(self, level, working_source, working_target, rhs_mesh, lambda_mesh)
    return wrapper


class SmoothSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return {
            param: arg for param, arg in zip(params, args)
        }

    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        layers = [
            ParamStripper(('self',)),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            RowMajorInteriorPoints(subconfig),
            PyBasicConversions(),
            LayerPrinter(),
        ]
        func = apply_all_layers(layers, func)

    def finalize(self, transform_result, program_config):
        pass