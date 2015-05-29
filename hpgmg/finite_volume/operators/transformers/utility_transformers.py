import ast
import sys
from hpgmg.finite_volume.operators.nodes import ArrayIndex

__author__ = 'nzhang-dev'


class SelfStripper(ast.NodeTransformer):
    def visit_arguments(self, node):
        if sys.version_info.major == 2:
            node.args = [
                arg for arg in node.args if arg.id != 'self'
            ]
        return node

class IndexTransformer(ast.NodeTransformer):
    def __init__(self, indices=()):
        self.indices = indices

    def visit_Name(self, node):
        if node.id in self.indices:
            return ArrayIndex(node.id)
        return node


class IndexOpTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.left, ArrayIndex):
            