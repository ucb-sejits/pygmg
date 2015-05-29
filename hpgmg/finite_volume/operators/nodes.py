from collections import Iterable

__author__ = 'nzhang-dev'

import ast

class PygmgNode(ast.AST):
    pass

class PygmgSemanticNode(PygmgNode):
    pass

class PyComprehension(PygmgSemanticNode):
    _fields = ['elts']

    def __init__(self, elts=None):
        self.elts = elts if elts is not None else []

class ArrayIndex(PygmgSemanticNode):
    _fields = ['name']
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        assert isinstance(other, (ast.Tuple, ast.List))

