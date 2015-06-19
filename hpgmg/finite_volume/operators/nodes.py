import copy

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

    def __deepcopy__(self, memo):
        return ArrayIndex(self.name)

class Parentheses(PygmgNode):
    _fields = ['value']
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "({})".format(str(self.value))

class RangeNode(PygmgSemanticNode):
    _fields = ['target', 'body']
    def __init__(self, target, iterator, body):
        self.iterator = iterator
        self.target = target
        self.body = body

    def __deepcopy__(self, memo):
        return RangeNode(self.target, self.iterator, copy.deepcopy(self.body))