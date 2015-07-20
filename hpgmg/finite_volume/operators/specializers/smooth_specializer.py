from __future__ import division, print_function
import ast
import atexit
import ctypes
import inspect
import math
from ast import Name

from ctree.c.nodes import SymbolRef, CFile, FunctionCall, ArrayDef, Array, For, String, Assign, Constant, Lt, PostInc
from ctree.cpp.nodes import CppInclude
from ctree.tune import MinimizeTime
import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
import time
import hpgmg
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction

from hpgmg.finite_volume.operators.specializers.util import to_macro_function, apply_all_layers, include_mover, \
    LayerPrinter, time_this
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.semantic_transformer import SemanticFinder
from hpgmg.finite_volume.operators.transformers.semantic_transformers.csemantics import CRangeTransformer
from hpgmg.finite_volume.operators.transformers.semantic_transformers.ompsemantics import OmpRangeTransformer
from hpgmg.finite_volume.operators.transformers.transformer_util import nest_loops
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer, FunctionCallTimer
from hpgmg.finite_volume.operators.tune.tuners import SmoothTuningDriver


__author__ = 'nzhang-dev'


class SmoothCFunction(PyGMGConcreteSpecializedFunction):

    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     #print("SmoothCFunction Finalize", entry_point_name)
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        thing, level, working_source, working_target, rhs_mesh, lambda_mesh = args
        c_args = [
            working_source, working_target,
            rhs_mesh, lambda_mesh
        ] + level.beta_face_values + [level.alpha]
        return c_args, {}

    # def __call__(self, thing, level, working_source, working_target, rhs_mesh, lambda_mesh):
    #     args = [
    #         working_source, working_target,
    #         rhs_mesh, lambda_mesh
    #     ] + level.beta_face_values + [level.alpha]
    #     # args.extend(level.beta_face_values)
    #     # args.append(level.alpha)
    #     #flattened = [arg.ravel() for arg in args]
    #     #self._c_function(*flattened)
    #     self._c_function(*args)

class CSmoothSpecializer(LazySpecializedFunction):

    #argspec = ['source', 'target', 'rhs_mesh', 'lambda_mesh', '*level.beta_face_values', 'level.alpha']

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

    def get_tuning_driver(self):
        if hpgmg.finite_volume.CONFIG.tune:
            return SmoothTuningDriver(objective=MinimizeTime())
        return super(CSmoothSpecializer, self).get_tuning_driver()

    def args_to_subconfig(self, args):
        params = (
            'self', 'level', 'source',
            'target', 'rhs_mesh', 'lambda_mesh'
        )
        return self.SmoothSubconfig({
            param: arg for param, arg in zip(params, args)
        })

    RangeTransformer = CRangeTransformer

    #@time_this
    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner = program_config
        #print(tuner, end="\t")
        ndim = subconfig['self'].operator.solver.dimensions
        ghost = subconfig['self'].operator.ghost_zone
        subconfig['ghost'] = ghost
        shape = subconfig['level'].interior_space
        layers = [
            ParamStripper(('self', 'level')),
            AttributeRenamer({
                'self.operator.apply_op': Name('apply_op', ast.Load())
            }),
            SemanticFinder(subconfig),
            self.RangeTransformer(cache_hierarchy=tuner),
            #RowMajorInteriorPoints(subconfig),
            AttributeGetter({'self': subconfig['self']}),
            ArrayRefIndexTransformer(
                encode_map={
                    'index': 'encode'
                },
                ndim=ndim
            ),
            PyBasicConversions(),
            #FunctionCallTimer((self.original_tree.body[0].name,)),
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
        # if subconfig['self'].operator.is_variable_coefficient:
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
        #if subconfig['self'].operator.solver.is_helmholtz:
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
        #print("codegen")
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = SmoothCFunction()
        subconfig = program_config[0]
        param_types = [np.ctypeslib.ndpointer(thing.dtype, len(thing.shape), thing.shape) for thing in
        [
            subconfig[key] for key in ('source', 'target', 'rhs_mesh', 'lambda_mesh')
        ]]
        beta_sample = subconfig['level'].beta_face_values[0]
        beta_type = np.ctypeslib.ndpointer(
            beta_sample.dtype,
            len(beta_sample.shape),
            beta_sample.shape
        )
        #if subconfig['self'].operator.is_variable_coefficient:
        param_types.extend(
            [beta_type]*subconfig['self'].operator.dimensions
        )
        #if subconfig['self'].operator.solver.is_helmholtz:
        param_types.append(
            param_types[-1]
        )  # add 1 more for alpha
        #print(dump(self.original_tree))
        name = self.tree.body[0].name
        return fn.finalize(name, Project(transform_result),
                           ctypes.CFUNCTYPE(None, *param_types))

    def __call__(self, *args, **kwargs):
        if hpgmg.finite_volume.CONFIG.tune:
            tune_count = 0
            tune_time = time.time()
            while not self._tuner.is_exhausted():
                #super(CSmoothSpecializer, self).__call__(*args, **kwargs)  # for the cache/codegen
                t = time.time()
                super(CSmoothSpecializer, self).__call__(*args, **kwargs)
                total_time = time.time() - t
                #cprint(total_time)
                self.report(time=total_time)
                tune_count += 1
            tune_time = time.time() - tune_time
            if tune_count:
                def print_report():
                    subconfig = self.args_to_subconfig(args)['level'].interior_space
                    print(subconfig)
                    print("Function:", type(self).__name__, "tuning time:", tune_time,
                          "tune count:", tune_count,
                          "best config:",
                          self._tuner.best_configs[subconfig])
                atexit.register(print_report)
        res = super(CSmoothSpecializer, self).__call__(*args, **kwargs)
        return res



class OmpSmoothSpecializer(CSmoothSpecializer):

    RangeTransformer = OmpRangeTransformer

    def transform(self, tree, program_config):
        stuff = super(OmpSmoothSpecializer, self).transform(tree, program_config)
        stuff[0].config_target = 'omp'
        stuff[0].body.insert(0, CppInclude("omp.h"))
    #     for_loop = stuff[0].find(For)
    #     subconfig = program_config[0]
    #     ndim = subconfig['self'].operator.solver.dimensions
    #     for_loop.pragma = 'omp parallel for private(a_x, b)'.format(ndim)
    #     # last_loop = list(stuff[0].find_all(For))[-1]
    #     # last_loop.body.append(
    #     #     FunctionCall(
    #     #         SymbolRef("printf"),
    #     #         args=[String(r"Threads: %d\n"), FunctionCall(SymbolRef("omp_get_num_threads"))]
    #     #     )
    #     # )
        return stuff