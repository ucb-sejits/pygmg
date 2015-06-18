import ast
from ctree.transformations import PyBasicConversions
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeGetter, \
    LookupSimplificationTransformer, AttributeRenamer

__author__ = 'nzhang-dev'
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import dump, get_ast

__author__ = 'nzhang-dev'


class CRestrictSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        return {
            'self': args[0],
            'level': args[1],
            'target': args[2],
            'source': args[3],
            'restriction_type': args[4]
        }
    def transform(self, tree, program_config):
        #print(dump(tree))
        subconfig, tuner_config = program_config
        layers = [
            ParamStripper(('self', 'level', 'restriction_type')),
            SemanticFinder(subconfig, locals=subconfig),
            AttributeRenamer({'restriction_type': ast.Num(n=subconfig['restriction_type'])}),
            AttributeGetter(subconfig),
            LookupSimplificationTransformer(),
            #PyBasicConversions()
        ]
        tree = apply_all_layers(layers, tree)
        print(dump(tree))