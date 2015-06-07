import ast
import sys
from ctree.c.nodes import FunctionCall, SymbolRef, ArrayRef, Constant, Add, Sub, Number
from ctree.types import get_ctype
from hpgmg.finite_volume.operators.nodes import ArrayIndex

__author__ = 'nzhang-dev'


def get_name(attribute_node):
    if isinstance(attribute_node, ast.Name):
        return attribute_node.id
    return get_name(attribute_node.value) + "." + attribute_node.attr

class ParamStripper(ast.NodeTransformer):
    def __init__(self, params):
        self.params = params

    def visit_arguments(self, node):
        if sys.version_info.major == 2:
            node.args = [
                arg for arg in node.args if arg.id not in self.params
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
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if isinstance(node.op, ast.Add):
                    op = Add
                else:
                    op = Sub
                ndim = len(node.right.elts)
                args = [
                    op(ArrayRef(SymbolRef(node.left.name), Constant(i)),
                             node.right.elts[i]) for i in range(ndim)
                ]
                return FunctionCall(
                    func=SymbolRef('encode'),
                    args=args
                )
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node


class IndexDirectTransformer(ast.NodeTransformer):
    def __init__(self, ndim):
        self.ndim = ndim

    def visit_ArrayIndex(self, node):
        return FunctionCall('encode', args=[
            ArrayRef(SymbolRef(node.name), Constant(i)) for i in range(self.ndim)
        ])


class AttributeRenamer(ast.NodeTransformer):
    def __init__(self, substitutes):
        self.substitutes = substitutes

    def visit_Attribute(self, node):
        name = get_name(node)
        if name in self.substitutes:
            return self.substitutes[name]
        return node

class AttributeGetter(ast.NodeTransformer):
    def __init__(self, namespace):
        self.namespace = namespace

    def get_value(self, node):
        name = get_name(node)
        attributes = name.split('.')
        error = AttributeError("Could not find {} in {}".format(name, self.namespace))
        if attributes[0] in self.namespace:
            obj = self.namespace[attributes.pop(0)]
            try:
                while attributes:
                    obj = getattr(obj, attributes.pop(0))
            except AttributeError:
                raise error
            return obj
        raise error

    def visit_Attribute(self, node):
        try:
            val = self.get_value(node)
            return Number(val, get_ctype(val))
        except AttributeError:
            return node

class ArrayRefIndexTransformer(ast.NodeTransformer):
    def __init__(self, indices, encode_func_name, ndim):
        self.indices = indices
        self.encode_func_name = encode_func_name
        self.ndim = ndim

    def visit_Index(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.indices:
            node.value = ast.Call(
                func=ast.Name(self.encode_func_name, ast.Load()),
                args=[
                    ast.Subscript(
                        value=ast.Name(id=node.value.id, ctx=ast.Load()),
                        slice=ast.Index(
                            ast.Num(n=dim)
                        ),
                        ctx=ast.Load()
                    ) for dim in range(self.ndim)
                ],
                keywords=[],
                starargs=None,
                kwargs=None,
            )
        return node