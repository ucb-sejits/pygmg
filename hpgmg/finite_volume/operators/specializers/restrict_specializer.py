import ast
import ctypes
from ctree.c.nodes import Assign, For, SymbolRef, Constant, PostInc, Lt, FunctionDecl, CFile
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover, time_this
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeGetter, \
    LookupSimplificationTransformer, AttributeRenamer, FunctionCallSimplifier, IndexTransformer, LoopUnroller, \
    IndexOpTransformer, IndexDirectTransformer, IndexOpTransformBugfixer, PyBranchSimplifier

import numpy as np

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction

__author__ = 'nzhang-dev'

class RestrictCFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, target, source, restriction_type):
        #print(self.entry_point_name, [i.shape for i in flattened])
        self._c_function(target.ravel(), source.ravel())


class CRestrictSpecializer(LazySpecializedFunction):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['{}_{}'.format(node.target, i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index, sym_type=ctypes.c_uint64()), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            bottom.body = node.body
            self.generic_visit(bottom)
            return top

    class RestrictSubconfig(dict):
        def __hash__(self):
            hash_thing = (
                self['level'].space,
                self['level'].ghost_zone,
                self['self'].neighbor_offsets,
                self['restriction_type']
            )
            #print(hash_thing)
            return hash(hash_thing)

    def args_to_subconfig(self, args):
        return self.RestrictSubconfig({
            'self': args[0],
            'level': args[1],
            'target': args[2],
            'source': args[3],
            'restriction_type': args[4]
        })


    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        ndim = subconfig['self'].dimensions
        layers = [
            ParamStripper(('self', 'level', 'restriction_type')),
            AttributeRenamer({'restriction_type': ast.Num(n=subconfig['restriction_type'])}),
            PyBranchSimplifier(),
            SemanticFinder(subconfig, locals=subconfig),
            IndexTransformer(('target_point', 'source_point')),
            AttributeGetter(subconfig),

            LookupSimplificationTransformer(),
            FunctionCallSimplifier(),
            LoopUnroller(),
            IndexOpTransformer(ndim=ndim),
            IndexDirectTransformer(ndim=ndim),
            self.RangeTransformer(),
            IndexOpTransformBugfixer(),
            PyBasicConversions(),
        ]
        tree = apply_all_layers(layers, tree)
        function = tree.find(FunctionDecl)
        function.defn = [
            SymbolRef('source_point_{}'.format(i), sym_type=ctypes.c_uint64())
            for i in range(ndim)
        ] + function.defn
        function.name = 'totally_not_restrict'
        for param in function.params:
            param.type = ctypes.POINTER(ctypes.c_double)()
        #print(dump(tree))
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        cfile = CFile(body=[tree, encode_func])
        cfile = include_mover(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = RestrictCFunction()
        subconfig, tuner_config = program_config
        np_arrays = subconfig['target'], subconfig['source']
        param_types = [
            np.ctypeslib.ndpointer(arr.dtype, 1, arr.size) for arr in np_arrays
        ]
        name = self.original_tree.body[0].name
        return fn.finalize('totally_not_restrict', Project(transform_result),
                           ctypes.CFUNCTYPE(None, *param_types))