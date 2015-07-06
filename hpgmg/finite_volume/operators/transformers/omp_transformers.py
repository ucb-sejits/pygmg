import ast
from ctree.c.nodes import SymbolRef, For, Assign, Constant, Lt, PostInc
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops

__author__ = 'nzhang-dev'

class RangeTransformer(ast.NodeTransformer):
        def __init__(self, block_hierarchy):
            """
            :param block_hierarchy: step sizes in each dimension
            :return:
            """
            self.block_hierarchy = block_hierarchy

        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['index_{}'.format(i) for i in range(ndim)]
            num_blocked = len(self.block_hierarchy)
            block_names = ['block_{}'.format(i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            top.pragma = 'omp parallel for collapse(2)'
            bottom.body = node.body
            self.generic_visit(bottom)
            return top