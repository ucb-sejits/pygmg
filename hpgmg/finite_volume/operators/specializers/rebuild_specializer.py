import ctypes
import math
import ast

from ctree.c.nodes import SymbolRef, ArrayDef, Array, CFile, FunctionCall
from ctree.cpp.nodes import CppInclude, CppDefine
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.transformations import PyBasicConversions
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
import numpy as np

from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, CompReductionTransformer
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer, IndexOpTransformer, LookupSimplificationTransformer, IndexTransformer, \
    IndexDirectTransformer, BranchSimplifier


__author__ = 'nzhang-dev'

class RebuildCFunction(ConcreteSpecializedFunction):
    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, thing, target_level):
        args = [target_level.valid, target_level.l1_inverse, target_level.d_inverse]
        if thing.is_variable_coefficient:
            args.extend(target_level.beta_face_values)
            if thing.solver.is_helmholtz:
                args.append(target_level.alpha)
        flattened = [arg.ravel() for arg in args]
        return self._c_function(*flattened)

class CRebuildSpecializer(LazySpecializedFunction):

    class RebuildSpecializerSubconfig(dict):
        def __hash__(self):
            things_to_hash = []
            target_level_things = ('l1_inverse', 'd_inverse', 'valid', 'alpha')
            for key in target_level_things:
                things_to_hash.append(
                    getattr(self['target_level'], key).shape
                )
            self_things = ('h2inv', 'a', 'b')
            for key in self_things:
                things_to_hash.append(
                    getattr(self['self'], key)
                )
            things_to_hash.extend(self['self'].neighborhood_offsets)

            return hash(tuple(things_to_hash))

    def args_to_subconfig(self, args):
        return self.RebuildSpecializerSubconfig({
            'self': args[0], 'target_level': args[1]
        })

    def transform(self, tree, program_config):
        func = tree.body[0]
        subconfig, tuner_config = program_config
        subconfig['ghost'] = subconfig['self'].ghost_zone
        ndim = subconfig['self'].dimensions
        #print(dump(tree))
        layers = [
            ParamStripper(('self',)),
            RowMajorInteriorPoints(subconfig),
            GeneratorTransformer(subconfig),
            CompReductionTransformer(),
            AttributeRenamer({
                'target_level.l1_inverse': ast.Name('l1_inverse', ast.Load()),
                'target_level.d_inverse': ast.Name('d_inverse', ast.Load()),
                'target_level.valid': ast.Name('valid', ast.Load()),
                'target_level.alpha': ast.Name('alpha', ast.Load()),
                'target_level.beta_face_values': ast.Name('beta_face_values', ast.Load()),
            }),
            AttributeGetter(subconfig),
            IndexTransformer(('index',)),
            ArrayRefIndexTransformer(
                indices=['index'],
                encode_func_name='encode',
                ndim=ndim
            ),
            LookupSimplificationTransformer(),
            IndexOpTransformer(),
            IndexDirectTransformer(ndim),
            PyBasicConversions(constants_dict={'False': 0, 'True': 1}),
            BranchSimplifier()
            #LayerPrinter(),
        ]
        func = apply_all_layers(layers, func)
        type_decls = {
            'adjust_value': ctypes.c_double(),
            'dominant_eigenvalue': ctypes.c_double(),
            'sum_abs': ctypes.c_double(),
            'a_diagonal': ctypes.c_double(),
            '____temp__sum_abs': ctypes.c_double(),
            '____temp__a_diagonal': ctypes.c_double()
        }
        func.params.extend(
            SymbolRef(name) for name in ('valid', 'l1_inverse', 'd_inverse')
        )
        if subconfig['self'].is_variable_coefficient:
            func.params.extend([
                SymbolRef("beta_face_values_{}".format(i)) for i in range(ndim)
            ])
            if subconfig['self'].solver.is_helmholtz:
                func.params.append(
                    SymbolRef("alpha")
                )
        params = []
        for param in func.params:
            if param.name == 'target_level':
                continue
            param.type = ctypes.POINTER(ctypes.c_double)()
            params.append(param)
        func.params = params
        beta_def = ArrayDef(
            SymbolRef('beta_face_values', sym_type=ctypes.POINTER(ctypes.c_double)()),
            size=ndim,
            body=Array(body=[
                SymbolRef("beta_face_values_{}".format(i)) for i in range(ndim)
            ])
        )
        defn = [
            SymbolRef(name, sym_type=t) for name, t in type_decls.items()
        ]
        if subconfig['self'].is_variable_coefficient:
            defn.append(beta_def)

        func.defn = defn + func.defn
        func.return_type = ctypes.c_double()
        ordering = Ordering([MultiplyEncode()])
        bits_per_dim = min([math.log(i, 2) for i in subconfig['target_level'].space]) + 1
        encode_func = ordering.generate(ndim, bits_per_dim, ctypes.c_uint64)

        cfile = CFile(body=[
            func,
            encode_func,
            CppInclude('stdint.h'),
            CppInclude('math.h'),
            CppDefine('abs', ['x'], FunctionCall(SymbolRef('fabs'), [SymbolRef('x')]))
        ])
        cfile = include_mover(cfile)
        #print(cfile)
        return [cfile]

    def finalize(self, transform_result, program_config):
        fn = RebuildCFunction()
        subconfig = program_config[0]
        valid = subconfig['target_level'].valid
        parameter_types = [
            np.ctypeslib.ndpointer(
                valid.dtype,
                1,
                valid.size
            )
        ]
        copies = 3
        if subconfig['self'].is_variable_coefficient:
            copies += subconfig['self'].dimensions
            if subconfig['self'].solver.is_helmholtz:
                copies += 1
        parameter_types *= copies
        name = self.original_tree.body[0].name
        return fn.finalize(
            name, Project(transform_result),
            ctypes.CFUNCTYPE(ctypes.c_double, *parameter_types)
        )
