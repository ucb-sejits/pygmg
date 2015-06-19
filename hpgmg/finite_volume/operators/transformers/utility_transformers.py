from __future__ import print_function
import ast
import copy
import sys
from ctree.frontend import dump

from ctree.c.nodes import FunctionCall, SymbolRef, Add, Sub, Constant, MultiNode, Div, Mul, BinaryOp, Assign, Op

from hpgmg.finite_volume.operators.nodes import ArrayIndex


__author__ = 'nzhang-dev'

def to_node(obj):
    from hpgmg.finite_volume.operators.transformers.generator_transformers import to_node as func
    return func(obj)

def get_name(attribute_node):
    if isinstance(attribute_node, ast.Name):
        return attribute_node.id
    return get_name(attribute_node.value) + "." + attribute_node.attr


def eval_node(node, locals, globals):
    expr = ast.Expression(node)
    expr = ast.fix_missing_locations(expr)
    #print(locals, globals)
    items = eval(
        compile(expr, "<string>", "eval"),
        globals, locals
    )
    return items

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
        if isinstance(node.left, (ast.Tuple, ast.List)) and isinstance(node.right, (ast.Tuple, ast.List)):
            return self.generic_visit(
                ast.Tuple(
                    elts=[
                        ast.BinOp(left=a, right=b, op=op) for a, b in zip(node.left.elts, node.right.elts)
                    ],
                    ctx=ast.Load()
                )
            )
        return node

class IndexOpTransformBugfixer(ast.NodeTransformer):
    """
    Designed to fix the Index = Index + Things encoding bug
    """
    def __init__(self, func_name='encode'):
        self.func_name = func_name

    def visit_Assign(self, node):
        target = node.targets[0]
        if isinstance(target, ast.Call) and isinstance(node.value, ast.Call):
            if target.func.id == node.value.func.id == self.func_name:
                return MultiNode([
                    Assign(a, b) for a, b in zip(target.args, node.value.args)
                ])
        return self.generic_visit(node)


class IndexDirectTransformer(ast.NodeTransformer):
    def __init__(self, ndim, encode_func_name='encode'):
        self.ndim = ndim
        self.encode_func_name = encode_func_name

    def visit_ArrayIndex(self, node):
        return ast.Call(func=ast.Name(id=self.encode_func_name, ctx=ast.Load()), args=[
            ast.Name(id=node.name+"_{}".format(i), ctx=ast.Load()) for i in range(self.ndim)
        ],
                        keywords=None, starargs=None)


class AttributeRenamer(ast.NodeTransformer):
    def __init__(self, substitutes):
        self.substitutes = substitutes

    def visit_Name(self, node):
        return self.visit_Attribute(node)

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
        #print(node)
        if isinstance(node.value, (ast.List, ast.Tuple)):
            #print(dump(node))
            #print("FOUND")
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

class FunctionCallSimplifier(ast.NodeTransformer):
    def visit_Call(self, node):
        if node.func.id == 'len':
            return ast.Num(n=len(node.args[0].elts))
        return self.generic_visit(node)

class LoopUnroller(ast.NodeTransformer):
    def visit_For(self, node):
        body = node.body
        result = MultiNode()
        for elt in node.iter.elts:
            #print(*[dump(i) for i in body])
            body_copy = [AttributeRenamer({node.target.id: elt}).visit(i) for i in copy.deepcopy(body)]
            body_copy = [self.visit(i) for i in body_copy]
            result.body.extend(body_copy)
        return result

class PyBranchSimplifier(ast.NodeTransformer):
    def visit_If(self, node):
        test = node.test
        try:
            result = eval_node(test, {}, {})
            if result:
                return self.visit(MultiNode(body=node.body))
            else:
                return self.visit(MultiNode(body=node.orelse))
        except:
            return self.generic_visit(node)