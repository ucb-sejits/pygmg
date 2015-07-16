import ast
import ctypes
from ctree.c.nodes import Assign, For, SymbolRef, Constant, Lt, PostInc, CFile
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction
from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import AttributeRenamer, AttributeGetter, \
    ParamStripper, ArrayRefIndexTransformer, IndexOpTransformer, IndexTransformer, IndexDirectTransformer, \
    IndexOpTransformBugfixer

import numpy as np

from ctree.frontend import dump

__author__ = 'nzhang-dev'

class InterpolateCFunction(PyGMGConcreteSpecializedFunction):
    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        target_mesh, source_mesh = args[-2:]
        return (target_mesh.ravel(), source_mesh.ravel()), {}

    # def __call__(self, thing, target_level, target_mesh, source_mesh):
    #     args = [
    #         target_mesh,
    #         source_mesh
    #     ]
    #     flattened = [i.ravel() for i in args]
    #     return self._c_function(*flattened)


class CInterpolateSpecializer(LazySpecializedFunction):

    def args_to_subconfig(self, args):
        return {
            'self': args[0],
            'target_level': args[1],
            'target_mesh': args[2],
            'source_mesh': args[3]
        }

    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['self'].dimensions
        #shape = subconfig['self'].operator.
        layers = [
            ParamStripper(('self', 'target_level')),
            # AttributeRenamer({
            #     'self.operator.apply_op': ast.Name('apply_op', ast.Load())
            # }),
            SemanticFinder(subconfig),
            AttributeGetter(subconfig),
            CRangeTransformer(),
            IndexTransformer(('target_index', 'source_index')),
            IndexOpTransformer(ndim, {'target_index': 'target_encode', 'source_index': 'source_encode'}),
            ArrayRefIndexTransformer(
                encode_map={
                    'target_index': 'target_encode',
                    'source_index': 'source_encode'
                },
                ndim=ndim
            ),
            IndexDirectTransformer(ndim, encode_func_names={'target_index': 'target_encode', 'source_index': 'source_encode'}),
            IndexOpTransformBugfixer(func_names=('target_encode', 'source_encode')),
            PyBasicConversions(),
        ]
        func = apply_all_layers(layers, func)
        for param in func.params:
            param.type = ctypes.POINTER(ctypes.c_double)()

        # func.defn = [
        #     SymbolRef('source_index', sym_type=ctypes.c_uint64()),
        #     SymbolRef('target_index', sym_type=ctypes.c_uint64())
        # ] + func.defn

        ordering = Ordering([MultiplyEncode()], prefix="source_")
        source_bits_per_dim = min([math.log(i, 2) for i in subconfig['source_mesh'].space]) + 1
        target_bits_per_dim = min([math.log(i, 2) for i in subconfig['target_mesh'].space]) + 1
        source_encode = ordering.generate(ndim, source_bits_per_dim, ctypes.c_uint64)
        ordering.prefix = 'target_'
        target_encode = ordering.generate(ndim, target_bits_per_dim, ctypes.c_uint64)

        # print(source_encode)
        # print(target_encode)
        # print(func)

        cfile = CFile(body=[
            source_encode,
            target_encode,
            func
        ])
        cfile = include_mover(cfile)
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = InterpolateCFunction()
        subconfig = program_config[0]
        name = self.tree.body[0].name
        ctype = [np.ctypeslib.ndpointer(
                subconfig[key].dtype,
                1,
                subconfig[key].size
            ) for key in ('target_mesh', 'source_mesh')]
        return fn.finalize(
            name,
            Project(transform_result),
            ctypes.CFUNCTYPE(None, *ctype)
        )