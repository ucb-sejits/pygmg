import ast
import ctypes
from ctree.c.nodes import SymbolRef, CFile, FunctionDecl, FunctionCall
from ctree.cpp.nodes import CppDefine
import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
import math
import rebox
from rebox.specializers.order import Ordering
from hpgmg.finite_volume.operators.specializers.util import to_macro_function, apply_all_layers, LayerPrinter, \
    validateCNode, include_mover
from ctree.frontend import dump, get_ast
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer
from rebox.specializers.rm.encode import MultiplyEncode
from ast import Name

__author__ = 'nzhang-dev'


def jit_smooth(func):

    specialized = SmoothSpecializer(get_ast(func))
    def wrapper(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        #to_macro_function(self.operator.apply_op)
        return specialized(self, level, working_source, working_target, rhs_mesh, lambda_mesh)
    return wrapper


class SmoothFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        return self

    def __call__(self, thing, level, working_source, working_target, rhs_mesh, lambda_mesh):
        args = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ]
        flattened = [arg.ravel() for arg in args]
        self._c_function(*flattened)

class SmoothSpecializer(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return {
            param: arg for param, arg in zip(params, args)
        }

    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['self'].operator.solver.dimensions
        #shape = subconfig['self'].operator.
        layers = [
            ParamStripper(('self', 'level')),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            RowMajorInteriorPoints(subconfig),
            AttributeGetter({'self': subconfig['self']}),
            ArrayRefIndexTransformer(
                indices=['index'],
                encode_func_name='encode',
                ndim=ndim
            ),
            PyBasicConversions(),
            #LayerPrinter(),
        ]
        func = apply_all_layers(layers, func)
        macro_func = to_macro_function(subconfig['self'].operator.apply_op)
        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space])
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        func.defn = [
            SymbolRef('a_x', sym_type=ctypes.c_float()),
            SymbolRef('b', sym_type=ctypes.c_float())
        ] + func.defn
        for call in func.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()
        for param in func.params:
            param.type = ctypes.POINTER(ctypes.c_float)()
        # print(func)
        # print(macro_func)
        # print(encode_func)
        cfile = CFile(body=[
            func, macro_func, encode_func
        ])
        cfile = include_mover(cfile)
        #print(type(cfile))
        #print(subconfig['level'].space)
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = SmoothFunction()
        param_types = [np.ctypeslib.ndpointer(thing.dtype, 1, np.multiply.reduce(thing.shape)) for thing in
        [
            program_config[0][key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        k = [
            program_config[0][key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]
        return fn.finalize("smooth_points", Project(transform_result),
                    ctypes.CFUNCTYPE(None, *param_types))