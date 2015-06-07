import ast
from hpgmg.finite_volume.operators.transformers.utility_transformers import get_name

__author__ = 'nzhang-dev'

class RowMajorInteriorPoints(ast.NodeTransformer):
    def __init__(self, namespace):
        self.namespace = namespace

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call):
            iter_name = get_name(node.iter.func)
            split = iter_name.split(".")
            if len(split) == 2 and split[0] in self.namespace and split[1] == 'interior_points':
                print("found it")
        return node