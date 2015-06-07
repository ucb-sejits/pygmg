from __future__ import print_function, division
import ast

from ctree import get_ast
from ctree.cpp.nodes import CppDefine
from ctree.frontend import dump
from ctree.c.nodes import SymbolRef, MultiNode
from ctree.transformations import PyBasicConversions
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, CompReductionVisitor, \
    AttributeFiller
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexTransformer, \
    IndexOpTransformer, IndexDirectTransformer

__author__ = 'nzhang-dev'


def to_macro_function(f):
    tree = get_ast(f).body[0]
    tree = ParamStripper(('self',)).visit(tree)
    name = SymbolRef(tree.name)
    params = [SymbolRef(n.id) for n in tree.args.args]
    self = f.im_self
    layers = [
        ParamStripper(('self',)),
        GeneratorTransformer({'self': self}),
        CompReductionVisitor(),
        AttributeFiller({'self': self}),
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
