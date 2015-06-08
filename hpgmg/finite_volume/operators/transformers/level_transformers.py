import ast
import ctypes
from ctree.c.nodes import ArrayDef, SymbolRef, Constant, Array, MultiNode, For, Assign, ArrayRef, Lt, PostInc, \
    FunctionCall, String
from hpgmg.finite_volume.operators.transformers.utility_transformers import get_name
from functools import reduce

__author__ = 'nzhang-dev'

class RowMajorInteriorPoints(ast.NodeTransformer):
    def __init__(self, namespace):
        self.namespace = namespace

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call):
            iter_name = get_name(node.iter.func)
            iteration_variable_name = node.target.id
            split = iter_name.split(".")
            if len(split) == 2 and split[0] in self.namespace and split[1] == 'interior_points':
                # building index thingy
                level = self.namespace[split[0]]
                dimensions = level.solver.dimensions
                index_declaration = ArrayDef(SymbolRef(iteration_variable_name, sym_type=ctypes.c_uint64()),
                                             size=dimensions, body=Array(body=[Constant(0) for i in range(dimensions)]))
                stencil = self.namespace['self']
                ghost = stencil.operator.ghost_zone
                #print(ghost)
                sizes = level.space
                minimums, maximums = ghost, sizes - ghost
                ret_val = MultiNode()
                for_loops = [
                    For(
                        init=Assign(
                            ArrayRef(
                                SymbolRef(iteration_variable_name),
                                Constant(dim)
                            ),
                            Constant(ghost[dim])
                        ),
                        test=Lt(
                            ArrayRef(
                                SymbolRef(iteration_variable_name),
                                Constant(dim)
                            ),
                            Constant(maximums[dim])
                        ),
                        incr=PostInc(
                           ArrayRef(
                                SymbolRef(iteration_variable_name),
                                Constant(dim)
                            )
                        )
                    ) for dim in range(dimensions)
                ]
                first_for = for_loops[0]
                reduce_func = lambda x, y: (x.body.append(y), y)[1]
                last_for = reduce(reduce_func, for_loops)
                elts = [ArrayRef(SymbolRef(iteration_variable_name), Constant(i)) for i in range(dimensions)]
                encoded = FunctionCall(SymbolRef("encode"), elts)
                last_for.body = [
                    self.visit(n) for n in node.body
                ]# + [FunctionCall(SymbolRef('printf'), [String(r"%zu, %zu, %zu\t%zu\n")] + elts + [encoded])]
                return MultiNode([
                    index_declaration,
                    first_for
                ])
        return node