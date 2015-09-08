import ast
import ctypes
import pycl as cl
import operator
from ctree.c.nodes import Assign, SymbolRef, Constant, FunctionCall, For, Lt, AddAssign, Add, MultiNode
from hpgmg.finite_volume.operators.specializers.util import compute_local_work_size, flattened_to_multi_index
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops

__author__ = 'dorthyluu'

class OclTilingRangeTransformer(ast.NodeTransformer):
    # def __init__(self, cache_hierarchy=()):
    #     self.cache_hierarchy = tuple(cache_hierarchy)

    def visit_RangeNode(self, node):
        ndim = len(node.iterator.ranges)
        ranges = node.iterator.ranges
        interior_space = tuple(r[1] - r[0] for r in ranges)
        ghost_zone = tuple(r[0] for r in ranges)

        # local_index = get_local_id(0)
        # setup local_index_0 -> local_index_n by using flattened_to_multi_index and adding in ghost zones
        # ndim for loops that iterate through global offset groups

        body = []
        loops = []
        assignments = []

        body.append(
            Assign(SymbolRef("local_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_local_id"), [Constant(0)]))
        )
        body.append(
            Assign(SymbolRef("group_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_group_id"), [Constant(0)]))
        )
        # body.extend([SymbolRef("{}_{}".format(node.target, d), ctypes.c_ulong()) for d in range(ndim)])

        device = cl.clGetDeviceIDs()[-1]
        local_size = compute_local_work_size(device, interior_space)
        single_work_dim = int(round(local_size ** (1/float(ndim))))  # maximize V to SA ratio of block
        local_work_shape = tuple(single_work_dim for _ in range(ndim))  # something like (8, 8, 8) if lws = 512
        local_indices = flattened_to_multi_index(SymbolRef("local_id"), local_work_shape, None, ghost_zone)
        global_work_dims = [interior_space[d] / local_work_shape[d] for d in range(ndim)]

        # for d in range(ndim):
        #     body.append(Assign(SymbolRef("local_id_{}".format(d), ctypes.c_ulong()), local_indices[d]))
        #
        global_indices = flattened_to_multi_index(SymbolRef("group_id"), global_work_dims, local_work_shape, None)
        # for d in range(ndim):
        #     body.append(Assign(SymbolRef("global_id_{}".format(d), ctypes.c_ulong()), global_indices[d]))

        # for d in range(ndim):
        #     loop = For(init=Assign(SymbolRef("global_id_{}".format(d), ctypes.c_ulong()), Constant(0)),
        #                test=Lt(SymbolRef("global_id_{}".format(d)), Constant(interior_space[d])),
        #                incr=AddAssign(SymbolRef("global_id_{}".format(d)), Constant(local_work_shape[d])))
        #     loops.append(loop)
        #
        # for d in range(ndim):
        #     assignments.append(Assign(SymbolRef("{}_{}".format(node.target, d)),
        #                               Add(SymbolRef("global_id_{}".format(d)), SymbolRef("local_id_{}".format(d)))))
        #
        # top, bottom = nest_loops(loops)
        # bottom.body = assignments + node.body
        # body.append(top)
        #
        for d in range(ndim):
            body.append(Assign(SymbolRef("{}_{}".format(node.target, d), ctypes.c_ulong()),
                               Add(global_indices[d], local_indices[d])))
                                      # Add(SymbolRef("global_id_{}".format(d)), SymbolRef("local_id_{}".format(d)))))

        body.extend(node.body)

        return MultiNode(body=body)
