import ast
from hpgmg.finite_volume.operators.nodes import RangeNode
from hpgmg.finite_volume.operators.transformers.transformer_util import get_obj
from hpgmg.finite_volume.operators.transformers.utility_transformers import get_name

__author__ = 'nzhang-dev'


class SemanticFinder(ast.NodeTransformer):
    registered = {
        'interior_points',
    }
    def __init__(self, namespace=None):
        self.namespace = {} if namespace is None else namespace

    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call):
            return self.generic_visit(node)
        iter_name = get_name(node.iter.func)

        iteration_variable_name = node.target.id
        split = iter_name.split(".")
        obj_name = split[0]
        func_name = split[-1]
        #print(obj_name, func_name)
        if func_name not in self.registered or obj_name not in self.namespace:
            return self.generic_visit(node)
        func_obj = get_obj(self.namespace, iter_name)
        return RangeNode(iteration_variable_name, func_obj(), node.body)