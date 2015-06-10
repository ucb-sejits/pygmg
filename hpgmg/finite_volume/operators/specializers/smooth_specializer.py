from __future__ import division, print_function
import ast
import ctypes
import math
from ast import Name

from ctree.c.nodes import SymbolRef, CFile, FunctionCall, ArrayDef, Array, For, String
from ctree.cpp.nodes import CppInclude
import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode

from hpgmg.finite_volume.operators.specializers.util import to_macro_function, apply_all_layers, include_mover
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer


__author__ = 'nzhang-dev'


class SmoothCFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, level, working_source, working_target, rhs_mesh, lambda_mesh):
        args = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ]
        if thing.operator.is_variable_coefficient:
            args.extend(level.beta_face_values)
            if thing.operator.solver.is_helmholtz:
                args.append(level.alpha)
        flattened = [arg.ravel() for arg in args]
        #print(self.entry_point_name, [i.shape for i in flattened])
        self._c_function(*flattened)

class CSmoothSpecializer(LazySpecializedFunction):

    class SmoothSubconfig(dict):
        def __hash__(self):
            operator = self['self'].operator
            hashed = [
                operator.a, operator.b, operator.h2inv,
                operator.dimensions, operator.is_variable_coefficient,
                operator.ghost_zone, tuple(operator.neighborhood_offsets)
            ]
            for i in ('source', 'target', 'rhs_mesh', 'lambda_mesh'):
                hashed.append(self[i].shape)
            #print(hashed)
            return hash(tuple(hashed))


    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return self.SmoothSubconfig({
            param: arg for param, arg in zip(params, args)
        })

    #@time_this
    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        ndim = subconfig['self'].operator.solver.dimensions
        ghost = subconfig['self'].operator.ghost_zone
        subconfig['ghost'] = ghost
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
        macro_func = to_macro_function(subconfig['self'].operator.apply_op,
                                       rename={"level.beta_face_values": SymbolRef("beta_face_values"),
                                               "level.alpha": SymbolRef("alpha")})
        macro_func.params = [
            param for param in macro_func.params if param.name != 'level'
        ]
        #print(macro_func)
        #raise Exception()
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)
        defn = func.defn
        func.defn = [
            SymbolRef('a_x', sym_type=ctypes.c_double()),
            SymbolRef('b', sym_type=ctypes.c_double())
        ]
        if subconfig['self'].operator.is_variable_coefficient:
            # needs beta values
            beta_sample = subconfig['level'].beta_face_values[0]
            beta_def = ArrayDef(
                SymbolRef("beta_face_values", sym_type=ctypes.POINTER(ctypes.c_double)()),
                size=ndim,
                body=Array(
                    body=[
                        SymbolRef("beta_face_values_{}".format(i)) for i in range(ndim)
                    ]
                )
            )
            func.defn.append(beta_def)
            func.params.extend([
                SymbolRef("beta_face_values_{}".format(i), sym_type=ctypes.POINTER(ctypes.c_double)())
                for i in range(ndim)
            ])
            if subconfig['self'].operator.solver.is_helmholtz:
                func.params.append(
                    SymbolRef("alpha", sym_type=ctypes.POINTER(ctypes.c_double)())
                )
        func.defn.extend(defn)
        for call in func.find_all(FunctionCall):
            if call.func.name == 'apply_op':
                call.args.pop()  #remove level
        for param in func.params:
            param.type = ctypes.POINTER(ctypes.c_double)()
        # print(func)
        # print(macro_func)
        # print(encode_func)
        #print(func)
        cfile = CFile(body=[
            func, macro_func, encode_func
        ])
        cfile = include_mover(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = SmoothCFunction()
        subconfig = program_config[0]
        param_types = [np.ctypeslib.ndpointer(thing.dtype, 1, thing.size) for thing in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_type = np.ctypeslib.ndpointer(
                beta_sample.dtype,
                1,
                beta_sample.size
            )
        if subconfig['self'].operator.is_variable_coefficient:
            param_types.extend(
                [beta_type]*subconfig['self'].operator.dimensions
            )
            if subconfig['self'].operator.solver.is_helmholtz:
                param_types.append(
                    param_types[-1]
                ) # add 1 more for alpha
        #print(dump(self.original_tree))
        name = self.original_tree.body[0].name
        return fn.finalize(name, Project(transform_result),
                    ctypes.CFUNCTYPE(None, *param_types))

class OmpSmoothSpecializer(CSmoothSpecializer):

    def transform(self, tree, program_config):
        stuff = super(OmpSmoothSpecializer, self).transform(tree, program_config)
        stuff[0].config_target = 'omp'
        stuff[0].body.insert(0, CppInclude("omp.h"))
        for_loop = stuff[0].find(For)
        subconfig = program_config[0]
        ndim = subconfig['self'].operator.solver.dimensions
        for_loop.pragma = 'omp parallel for collapse({}) private(a_x, b)'.format(ndim)
        # last_loop = list(stuff[0].find_all(For))[-1]
        # last_loop.body.append(
        #     FunctionCall(
        #         SymbolRef("printf"),
        #         args=[String(r"Threads: %d\n"), FunctionCall(SymbolRef("omp_get_num_threads"))]
        #     )
        # )
        return stuff
