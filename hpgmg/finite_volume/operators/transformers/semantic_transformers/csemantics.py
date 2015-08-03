import ast
import ctypes
from ctree.c.nodes import For, Assign, SymbolRef, Constant, PostInc, Lt, AddAssign, Add
import itertools
import hpgmg
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops

__author__ = 'nzhang-dev'

class CRangeTransformer(ast.NodeTransformer):
    def __init__(self, cache_hierarchy=()):
        self.cache_hierarchy = tuple(cache_hierarchy)

    def visit_RangeNode(self, node):
        ndim = len(node.iterator.ranges)
        cache_hierarchy = self.cache_hierarchy[:ndim]
        cache_hierarchy = (1,) * (ndim - len(cache_hierarchy)) + cache_hierarchy
        outer_loops = []
        blocked_loops = []
        assignments = []

        make_declare = lambda x: SymbolRef(x.name, sym_type=ctypes.c_long())

        for dim, ((low, high), block_size) in enumerate(
                itertools.izip_longest(node.iterator.ranges, cache_hierarchy, fillvalue=1)):
            inner_index = SymbolRef("{}_{}".format(node.target, dim))
            blocking_index_outer = SymbolRef("{}_{}_coarse".format(node.target, dim))
            blocking_index_inner = SymbolRef("{}_{}_fine".format(node.target, dim))

            if block_size != 1 and block_size < (high - low) and ((high - low) % block_size == 0):
                # make block size nice
                outer_loop = For(
                    init=Assign(make_declare(blocking_index_outer), Constant(low)),
                    test=Lt(blocking_index_outer, Constant(high)),
                    incr=AddAssign(blocking_index_outer, Constant(block_size))
                )
                inner_loop = For(
                    init=Assign(make_declare(blocking_index_inner), Constant(0)),
                    test=Lt(blocking_index_inner, Constant(block_size)),
                    incr=PostInc(blocking_index_inner)
                )
                outer_loops.append(outer_loop)
                blocked_loops.append(inner_loop)
                assignments.append(Assign(make_declare(inner_index), Add(blocking_index_outer, blocking_index_inner)))
            else:
                # no blocking
                inner_loop = For(
                    init=Assign(make_declare(inner_index), Constant(low)),
                    test=Lt(inner_index, Constant(high)),
                    incr=PostInc(inner_index)
                )
                blocked_loops.append(inner_loop)



        top, bottom = nest_loops(outer_loops + blocked_loops)
        bottom.body = assignments + node.body
        self.generic_visit(bottom)
        return top