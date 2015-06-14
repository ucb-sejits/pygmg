import ast
import sys

from ctree.c.nodes import FunctionCall, SymbolRef, Add, Sub, Constant, MultiNode, Div, Mul

from hpgmg.finite_volume.operators.nodes import ArrayIndex


__author__ = 'nzhang-dev'

def to_node(obj):
    from hpgmg.finite_volume.operators.transformers.generator_transformers import to_node as func
    return func(obj)

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

    def __init__(self, ndim, encode_func_name='encode'):
        self.ndim = ndim
        self.encode_func_name = encode_func_name

    def visit_BinOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        op = node.op
        if isinstance(node.left, ArrayIndex):
            if isinstance(node.right, (ast.List, ast.Tuple)):
                #print(dump(node))
                args = [
                    ast.BinOp(left=ast.Name(id=node.left.name+"_{}".format(i), ctx=ast.Load()),
                              right=node.right.elts[i],
                              op=op) for i in range(self.ndim)
                ]
                return ast.Call(
                    func=ast.Name(id=self.encode_func_name, ctx=ast.Load()),
                    args=args,
                    keywords=[],
                    starargs=None
                )
            elif isinstance(node.right, ast.Num):
                node.right = ast.Tuple(elts=[node.right]*self.ndim, ctx=ast.Load())
                return self.visit(node)
        if isinstance(node.left, ast.Call) and node.left.func.id == self.encode_func_name:
            #print("ENCODE")
            if isinstance(node.right, (ast.Tuple, ast.List)):
                args = [
                    ast.BinOp(left=arg,
                              op=op,
                              right=elt)
                    for arg, elt in zip(node.left.args, node.right.elts)
                ]
                return ast.Call(
                    func=ast.Name(id=self.encode_func_name, ctx=ast.Load()),
                    args=args,
                    keywords=[],
                    starargs=None
                )
            elif isinstance(node.right, ast.Num):
                node.right = ast.Tuple(elts=[node.right]*self.ndim, ctx=ast.Load())
                return self.visit(node)
        return node


class IndexDirectTransformer(ast.NodeTransformer):
    def __init__(self, ndim, encode_func_name='encode'):
        self.ndim = ndim
        self.encode_func_name = encode_func_name

    def visit_ArrayIndex(self, node):
        return FunctionCall(self.encode_func_name, args=[
            SymbolRef(node.name+"_{}".format(i)) for i in range(self.ndim)
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
            return to_node(obj)
        raise error

    def visit_Attribute(self, node):
        try:
            val = self.get_value(node)
            return val
        except (AttributeError, ValueError):
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
                    ast.Name(id=node.value.id+"_{}".format(dim), ctx=ast.Load())
                    for dim in range(self.ndim)
                ],
                keywords=[],
                starargs=None,
                kwargs=None,
            )
        return node

class LookupSimplificationTransformer(ast.NodeTransformer):
    def visit_Subscript(self, node):
        #print("visited")
        #print(dump(node))
        if isinstance(node.value, (ast.List, ast.Tuple)):
            #print(dump(node))
            if isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Num):
                index = node.slice.value.n
                return self.visit(node.value.elts[index])
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node


class BranchSimplifier(ast.NodeTransformer):
    """C Transformer"""
    def visit_If(self, node):
        if isinstance(node.cond, Constant):
            if node.cond.value:
                return self.visit(MultiNode(body=node.then))
            return self.visit(MultiNode(body=node.elze))
        node.then = [self.visit(i) for i in node.then]
        if node.elze:
            node.elze = [self.visit(i) for i in node.elze]
        return node