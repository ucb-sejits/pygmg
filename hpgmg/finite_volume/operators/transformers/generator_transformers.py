import copy
import functools
from hpgmg.finite_volume.operators.nodes import PyComprehension

__author__ = 'nzhang-dev'

import ast

def to_node(obj):
    return ast.parse(repr(obj)).body[0].value


class GeneratorTransformer(ast.NodeTransformer):
    def __init__(self, _locals=None, _globals=None):
        self.locals = _locals if _locals is not None else {}
        self.globals = _globals if _globals is not None else {}
    def visit_GeneratorExp(self, node):
        elt = node.elt
        comprehension = node.generators[0]
        target = comprehension.target
        iterable = comprehension.iter
        items = eval(
            compile(ast.Expression(iterable), "<string>", "eval"),
            self.globals, self.locals
            )
        output = PyComprehension()
        for item in items:
            cp = copy.deepcopy(elt)
            output.elts.append(
                NameSwapper({target.id: to_node(item)}).visit(cp)
            )
        return output

    def visit_ListComp(self, node):
        return self.visit_GeneratorExp(node)

class NameSwapper(ast.NodeTransformer):
    def __init__(self, namespace):
        self.namespace = namespace

    def visit_Name(self, node):
        if node.id in self.namespace:
            return self.visit(self.namespace[node.id])
        return node

class AttributeFiller(ast.NodeTransformer):
    def __init__(self, namespace):
        self.namespace = namespace

    def visit_Attribute(self, node):
        obj = self.namespace[node.value.id]
        return self.visit(to_node(getattr(obj, node.attr)))

class CompReductionVisitor(ast.NodeTransformer):
    mapping = {
        "sum": lambda x, y: ast.BinOp(left=x, right=y, op=ast.Add()),
    }

    def visit_Call(self, node):
        if node.func.id in self.mapping:
            return functools.reduce(self.mapping[node.func.id], [self.visit(i) for i in node.args[0].elts])
        node.args = [self.visit(arg) for arg in node.args]
        return node