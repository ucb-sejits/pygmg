import ast
import ctypes
import random
from ctree.c.nodes import Assign, For, SymbolRef, Constant, PostInc, Lt, FunctionDecl, CFile
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
import math
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, \
    CompReductionTransformer
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover, time_this
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeGetter, \
    LookupSimplificationTransformer, AttributeRenamer, FunctionCallSimplifier, IndexTransformer, LoopUnroller, \
    IndexOpTransformer, IndexDirectTransformer, IndexOpTransformBugfixer, PyBranchSimplifier

import numpy as np

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import dump

__author__ = 'nzhang-dev'

class RestrictCFunction(PyGMGConcreteSpecializedFunction):

    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    def pyargs_to_cargs(self, args, kwargs):
        return (args[2].ravel(), args[3].ravel()), {}

    # def __call__(self, thing, level, target, source, restriction_type):
    #     #print(self.entry_point_name, [i.shape for i in flattened])
    #     self._c_function(target.ravel(), source.ravel())


class CRestrictSpecializer(LazySpecializedFunction):


    class RestrictSubconfig(dict):
        #hash_count = 0
        #pass
        def __hash__(self):
            hash_thing = (
                self['level'].space,
                self['level'].ghost_zone,
                self['self'].neighbor_offsets,
                self['source'].shape,
                self['target'].shape,
                self['restriction_type'],
            )
            return hash(hash_thing)
            #print(hash_thing)
        #     #self.hash_count += 1
        #     return id(self)

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
            AttributeGetter({'self': subconfig['self']}),
            PyBranchSimplifier(),
            SemanticFinder(subconfig, locals=subconfig),
            IndexTransformer(('target_point', 'source_point')),
            AttributeGetter(subconfig),
            LookupSimplificationTransformer(),
            FunctionCallSimplifier(),
            LoopUnroller(),
            GeneratorTransformer(subconfig),
            CompReductionTransformer(),
            IndexOpTransformer(ndim=ndim, encode_func_names={'target_point': 'target_encode', 'source_point': 'source_encode'}),
            IndexDirectTransformer(ndim=ndim, encode_func_names={'source_point': 'source_encode', 'target_point': 'target_encode'}),
            IndexOpTransformBugfixer(func_names=('target_encode', 'source_encode')),
            CRangeTransformer(),
            PyBasicConversions(),
        ]
        tree = apply_all_layers(layers, tree)
        #print(dump(tree))
        function = tree.find(FunctionDecl)
        function.defn = [
            SymbolRef('source_point_{}'.format(i), sym_type=ctypes.c_uint64())
            for i in range(ndim)
        ] + function.defn
        function.name = 'totally_not_restrict'
        for param in function.params:
            param.type = ctypes.POINTER(ctypes.c_double)()
        #print(dump(tree))
        ordering = Ordering([MultiplyEncode()], prefix='source_')
        bits_per_dim = min([math.log(i, 2) for i in subconfig['source'].shape]) + 1
        encode_func_source = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        ordering = Ordering([MultiplyEncode()], prefix='target_')
        bits_per_dim = min([math.log(i, 2) for i in subconfig['target'].shape]) + 1
        encode_func_target = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        cfile = CFile(body=[tree, encode_func_source, encode_func_target])
        cfile = include_mover(cfile)
        #print(subconfig['self'].neighbor_offsets[subconfig['restriction_type']])
        # if subconfig['restriction_type'] == 0:
        #     print(cfile)
        #print(subconfig['target'].shape, subconfig['source'].shape)
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = RestrictCFunction()
        subconfig, tuner_config = program_config
        np_arrays = subconfig['target'], subconfig['source']
        param_types = [
            np.ctypeslib.ndpointer(arr.dtype, 1, arr.size) for arr in np_arrays
        ]
        name = self.tree.body[0].name
        return fn.finalize('totally_not_restrict', Project(transform_result),
                           ctypes.CFUNCTYPE(None, *param_types))