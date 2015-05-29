from collections import namedtuple
import ctree
from ctree.frontend import dump, get_ast
import sys
from ctree.transformations import PyBasicConversions
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, CompReductionVisitor, \
    AttributeFiller
from hpgmg.finite_volume.operators.transformers.utility_transformers import SelfStripper, IndexTransformer

__author__ = 'nzhang-dev'
import functools

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction

def jit_apply_op(func):
    specialized = Apply_Op_Specializer(py_ast=get_ast(func))
    @functools.wraps(func)
    def wrapper(self, mesh, index, level):
        return specialized(self, mesh, index, level)
    return wrapper


class Apply_Op_Specializer(LazySpecializedFunction):

    def args_to_subconfig(self, args):
        Data = namedtuple("Data", ["dimensions", "num_neighbors", "neighborhood_offsets"])
        return {
            'self': args[0],
            'shape': args[1].shape,
            'num_dims': len(args[2])
        }

    def transform(self, tree, program_config):
        args_subconfig, tuning_config = program_config
        layers = [
            SelfStripper(),
            GeneratorTransformer(args_subconfig),
            CompReductionVisitor(),
            AttributeFiller(args_subconfig),
            IndexTransformer(('index',)),
            #PyBasicConversions()
        ]
        for layer in layers:
            tree = layer.visit(tree)

        print(dump(tree))
        raise Exception()



    def finalize(self, transform_result, program_config):
        pass