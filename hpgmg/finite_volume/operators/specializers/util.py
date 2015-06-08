from __future__ import print_function, division
import ast
import atexit

from ctree import get_ast, ctree
from ctree.cpp.nodes import CppDefine
from ctree.frontend import dump
from ctree.c.nodes import SymbolRef, MultiNode, BinaryOp
from ctree.transformations import PyBasicConversions
import time
import sys
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, CompReductionVisitor, \
    AttributeFiller
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexTransformer, \
    IndexOpTransformer, IndexDirectTransformer

__author__ = 'nzhang-dev'


def to_macro_function(f, namespace=None):
    namespace = {} if namespace is None else namespace.copy()
    tree = get_ast(f).body[0]
    tree = ParamStripper(('self',)).visit(tree)
    name = SymbolRef(tree.name)
    params = [SymbolRef(n.id) for n in tree.args.args]
    self = f.im_self
    namespace.update({'self': self})

    layers = [
        ParamStripper(('self',)),
        GeneratorTransformer(namespace),
        CompReductionVisitor(),
        AttributeFiller(namespace),
        IndexTransformer(('index',)),
        IndexOpTransformer(),
        IndexDirectTransformer(self.solver.dimensions),
        PyBasicConversions()
    ]
    body = MultiNode(body=[apply_all_layers(layers, node) for node in tree.body]).body[0].value
    define = CppDefine(name='apply_op', params=params, body=body)
    return define

def apply_all_layers(layers, node):
    for layer in layers:
        node = layer.visit(node)
    return node

class LayerPrinter(ast.NodeVisitor):
    def visit(self, node):
        print(dump(node))
        return node

def validateCNode(node):
    for node in ast.walk(node):
        if not isinstance(node, ctree.c.nodes.CNode):
            print('FAILED:')
            print(dump(node))
            return False
    return True

def include_mover(node):
    includes = set()
    defines = set()
    class includeFinder(ast.NodeTransformer):
        def visit_CppInclude(self, node):
            includes.add(node)
        def visit_CppDefine(self, node):
            defines.add(node)

    includeFinder().visit(node)
    node.body = [
        include for include in includes
    ]+[
        define for define in defines
    ]+node.body
    return node


def time_this(func):
    timings = []
    def wrapper(*args, **kwargs):
        a = time.time()
        res = func(*args, **kwargs)
        timings.append(time.time() - a)
        return res
    wrapper.total_time = 0
    @atexit.register
    def dump_time():
        print(func.__name__, sum(timings))
    return wrapper