import ast
from collections import OrderedDict
import ctypes
from ctree.c.nodes import SymbolRef, Constant, PostInc, Lt, For, Assign, FunctionDecl, CFile, Return
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
from hpgmg.finite_volume.operators.specializers.smooth_specializer import apply_all_layers
from hpgmg.finite_volume.operators.specializers.util import include_mover
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops

from ctree.frontend import dump
import numpy as np
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexDirectTransformer, \
    IndexTransformer

__author__ = 'nzhang-dev'

class MeshOpCFunction(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, *args):
        flattened = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                flattened.append(arg.ravel())
            elif isinstance(arg, (int, float)):
                flattened.append(arg)
        return self._c_function(*flattened)


class MeshOpSpecializer(LazySpecializedFunction):

    class RangeTransformer(ast.NodeTransformer):
        def visit_RangeNode(self, node):
            ndim = len(node.iterator.ranges)
            index_names = ['{}_{}'.format(node.target, i) for i in range(ndim)]
            for_loops = [For(
                init=Assign(SymbolRef(index), Constant(low)),
                test=Lt(SymbolRef(index), Constant(high)),
                incr=PostInc(SymbolRef(index))
            ) for index, (low, high) in zip(index_names, node.iterator.ranges)]
            top, bottom = nest_loops(for_loops)
            bottom.body = node.body
            self.generic_visit(bottom)
            return top

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        ndim = subconfig['self'].solver.dimensions
        layers = [
            ParamStripper(('self')),
            SemanticFinder(subconfig),
            self.RangeTransformer(),
            IndexTransformer(('index')),
            IndexDirectTransformer(ndim, {'index': 'encode'}),
            PyBasicConversions()
        ]

        tree = apply_all_layers(layers, tree)

        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['self'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        tree.find(CFile).body.append(encode_func)
        tree = include_mover(tree)
        return [tree]


class CFillMeshSpecializer(MeshOpSpecializer):

    class MeshSubconfig(dict):
        def __hash__(self):
            return hash(tuple(self[key].shape for key in ('mesh',)))

    def args_to_subconfig(self, args):
        return self.MeshSubconfig({
            key: arg for key, arg in zip(('self', 'mesh', 'value'), args)
        })

    def transform(self, f, program_config):
        f = super(CFillMeshSpecializer, self).transform(f, program_config)[0]
        #print(f)
        func_decl = f.find(FunctionDecl)
        param_types = [ctypes.POINTER(ctypes.c_double)(), ctypes.c_double()]
        for param, t in zip(func_decl.params, param_types):
            param.type = t

        f = include_mover(f)

        return [f]

    def finalize(self, transform_result, program_config):
        subconfig, tuner_config = program_config
        tree = transform_result[0]
        retval = None
        entry_point = self.original_tree.body[0].name
        param_types = []
        for param in (('mesh', 'value')):
            if isinstance(subconfig[param], np.ndarray):
                arr = subconfig[param]
                param_types.append(np.ctypeslib.ndpointer(
                    arr.dtype, 1, arr.size
                ))
            elif isinstance(subconfig[param], (int, float)):
                param_types.append(ctypes.c_double)

        fn = MeshOpCFunction()
        return fn.finalize(
            entry_point, Project(transform_result), ctypes.CFUNCTYPE(retval, *param_types)
        )

class CGeneralizedSimpleMeshOpSpecializer(MeshOpSpecializer):

    class GeneralizedSubconfig(OrderedDict):
        def __hash__(self):
            to_hash = []
            for key, value in self.items():
                if isinstance(value, np.ndarray):
                    to_hash.append((key, value.shape))
            return hash(tuple(to_hash))

    def args_to_subconfig(self, args):
        argument_names = [arg.id for arg in self.original_tree.body[0].args.args]
        retval = self.GeneralizedSubconfig()
        for key, val in ((argument_name, arg) for argument_name, arg in zip(argument_names, args)):
            retval[key] = val
        return retval

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        f = super(CGeneralizedSimpleMeshOpSpecializer, self).transform(tree, program_config)[0]
        decl = f.find(FunctionDecl)
        params = []
        for param in decl.params:
            if not isinstance(subconfig[param.name], (int, float, np.ndarray)):
                continue
            if isinstance(subconfig[param.name], (int, float)):
                param.type = ctypes.c_double()
            else:
                param.type = ctypes.POINTER(ctypes.c_double)()
            params.append(param)
        decl.params = params
        if decl.find(Return):
            decl.return_type = ctypes.c_double()
        #print(f)
        return [f]


    def finalize(self, transform_result, program_config):
        fn = MeshOpCFunction()
        subconfig, tuner_config = program_config
        param_types = []
        for key, value in subconfig.items():
            if isinstance(value, (int, float)):
                param_types.append(ctypes.c_double)
            if isinstance(value, np.ndarray):
                param_types.append(np.ctypeslib.ndpointer(value.dtype, 1, value.size))

        name = self.original_tree.body[0].name
        if any(isinstance(i, ast.Return) for i in ast.walk(self.original_tree)):
            return_type = ctypes.c_double
        else:
            return_type = None

        return fn.finalize(
            name, Project(transform_result), ctypes.CFUNCTYPE(return_type, *param_types)
        )