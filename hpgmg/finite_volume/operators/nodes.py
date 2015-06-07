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

class Parentheses(PygmgNode):
    _fields = ['value']
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "({})".format(str(self.value))
