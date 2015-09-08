import ast
import ctypes
import pycl as cl
import operator
from ctree.c.nodes import Assign, SymbolRef, Constant, FunctionCall, For, Lt, AddAssign, Add, MultiNode
from hpgmg.finite_volume.operators.specializers.util import compute_local_work_size, flattened_to_multi_index

__author__ = 'dorthyluu'

class OclRangeTransformer(ast.NodeTransformer):

    def visit_RangeNode(self, node):
        ndim = len(node.iterator.ranges)
        ranges = node.iterator.ranges
        shape = tuple(r[1] - r[0] for r in ranges)
        offsets = tuple(r[0] for r in ranges)


        body = []
        # body.append(SymbolRef("global_id", ctypes.c_ulong()))
        body.append(Assign(SymbolRef("global_id", ctypes.c_ulong()),
                           FunctionCall(SymbolRef("get_global_id"), [Constant(0)])))

        indices = flattened_to_multi_index(SymbolRef("global_id"), shape, None, offsets)
        for d in range(ndim):
            body.append(Assign(SymbolRef("{}_{}".format(node.target, d), ctypes.c_ulong()), indices[d]))
        body.extend(node.body)
        return MultiNode(body=body)

class OclTilingRangeTransformer(ast.NodeTransformer):

    def visit_RangeNode(self, node):
        # note: tried to set global size = local size and have one thread do as much work as possible, was slower
        ndim = len(node.iterator.ranges)
        ranges = node.iterator.ranges
        shape = tuple(r[1] - r[0] for r in ranges)
        offsets = tuple(r[0] for r in ranges)

        device = cl.clGetDeviceIDs()[-1]
        local_size = compute_local_work_size(device, shape)
        single_work_dim = int(round(local_size ** (1/float(ndim))))  # maximize V to SA ratio of block
        local_work_shape = tuple(single_work_dim for _ in range(ndim))  # something like (8, 8, 8) if lws = 512
        global_work_dims = [shape[d] / local_work_shape[d] for d in range(ndim)]
        local_indices = flattened_to_multi_index(SymbolRef("local_id"), local_work_shape, None, offsets)
        global_indices = flattened_to_multi_index(SymbolRef("group_id"), global_work_dims, local_work_shape, None)

        body = [
            Assign(SymbolRef("local_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_local_id"), [Constant(0)])),
            Assign(SymbolRef("group_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_group_id"), [Constant(0)]))
        ]
        for d in range(ndim):
            body.append(Assign(SymbolRef("{}_{}".format(node.target, d), ctypes.c_ulong()),
                               Add(global_indices[d], local_indices[d])))

        body.extend(node.body)

        return MultiNode(body=body)
