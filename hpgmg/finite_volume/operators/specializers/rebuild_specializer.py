import ctypes
import math
import ast
from ctree.c.macros import NULL

from ctree.c.nodes import SymbolRef, ArrayDef, Array, CFile, FunctionCall, FunctionDecl, Assign, MultiNode, Constant, \
    ArrayRef, For, Return, Ref
from ctree.cpp.nodes import CppInclude, CppDefine
from ctree.jit import LazySpecializedFunction
from ctree.nodes import Project
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.transforms.declaration_filler import DeclarationFiller
from rebox.specializers.order import Ordering
from rebox.specializers.rm.encode import MultiplyEncode
import numpy as np
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.specializers.jit import PyGMGConcreteSpecializedFunction, KernelRunManager, \
    PyGMGOclConcreteSpecializedFunction

from hpgmg.finite_volume.operators.specializers.util import apply_all_layers, include_mover, \
    compute_largest_local_work_size, new_generate_control, flattened_to_multi_index
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, CompReductionTransformer
from hpgmg.finite_volume.operators.transformers.level_transformers import RowMajorInteriorPoints
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, AttributeRenamer, \
    AttributeGetter, ArrayRefIndexTransformer, IndexOpTransformer, LookupSimplificationTransformer, IndexTransformer, \
    IndexDirectTransformer, BranchSimplifier, OclFileWrapper

import operator
import pycl as cl


__author__ = 'nzhang-dev'

class RebuildCFunction(PyGMGConcreteSpecializedFunction):
    # def finalize(self, entry_point_name, project_node, entry_point_typesig):
    #     self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
    #     self.entry_point_name = entry_point_name
    #     return self

    @staticmethod
    def pyargs_to_cargs(c_args, kwargs):
        thing, target_level = c_args
        c_args = [target_level.valid, target_level.l1_inverse, target_level.d_inverse]
        if thing.is_variable_coefficient:
            c_args.extend(target_level.beta_face_values)
            if thing.solver.is_helmholtz:
                c_args.append(target_level.alpha)
        flattened = [arg.ravel() for arg in c_args]
        return flattened, {}
    #
    # def __call__(self, thing, target_level):
    #     args = [target_level.valid, target_level.l1_inverse, target_level.d_inverse]
    #     if thing.is_variable_coefficient:
    #         args.extend(target_level.beta_face_values)
    #         if thing.solver.is_helmholtz:
    #             args.append(target_level.alpha)
    #     flattened = [arg.ravel() for arg in args]
    #     return self._c_function(*flattened)


class RebuildOclFunction(PyGMGOclConcreteSpecializedFunction):

    def get_all_args(self, args, kwargs):
        args = args + self.extra_args
        thing, target_level, final_answer = args
        args_to_bufferize = [target_level.valid, target_level.l1_inverse, target_level.d_inverse, final_answer]
        return args_to_bufferize

    def reduced_value(self):
        kernel = self.kernels[0]
        ary, evt = cl.buffer_to_ndarray(self.queue, self.extra_args[-1].buffer.buffer, self.extra_args[-1])
        kernel.args[0].dirty = False
        evt.wait()
        return self.extra_args[-1][0]


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
                encode_map={
                    'index': 'encode'
                },
                ndim=ndim
            ),
            LookupSimplificationTransformer(),
            IndexOpTransformer(ndim, encode_func_names={'index': 'encode'}),
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
        name = self.tree.body[0].name
        return fn.finalize(
            name, Project(transform_result),
            ctypes.CFUNCTYPE(ctypes.c_double, *parameter_types)
        )

class OclRebuildSpecializer(CRebuildSpecializer):

    def args_to_subconfig(self, args):
        return CRebuildSpecializer.RebuildSpecializerSubconfig({
            'self': args[0], 'target_level': args[1]
        })

    class RangeTransformer(ast.NodeTransformer):

        def __init__(self, subconfig):
            self.subconfig = subconfig
            super(OclRebuildSpecializer.RangeTransformer, self).__init__()

        def visit_For(self, node):
            target_level = self.subconfig['target_level']

            offsets = target_level.ghost_zone
            shape = target_level.interior_space

            current_body = node.body
            while(len(current_body) == 1):
                current_body = current_body[0].body

            body=[
                Assign(SymbolRef("global_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))
            ]
            indices = flattened_to_multi_index(SymbolRef("global_id"), shape, offsets=offsets)
            for d in range(len(shape)):
                body.append(Assign(SymbolRef("index_%d"%d, ctypes.c_ulong()), indices[d]))
            body.extend(current_body)
            return MultiNode(body=body)

    def transform(self, tree, program_config):

        func = tree.body[0]
        subconfig, tuner_config = program_config
        subconfig['ghost'] = subconfig['self'].ghost_zone
        ndim = subconfig['self'].dimensions
        layers = [
            ParamStripper(('self',)),
            self.RangeTransformer(subconfig),
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
                encode_map={
                    'index': 'encode'
                },
                ndim=ndim
            ),
            LookupSimplificationTransformer(),
            IndexOpTransformer(ndim, encode_func_names={'index': 'encode'}),
            IndexDirectTransformer(ndim),
            PyBasicConversions(constants_dict={'False': 0, 'True': 1}),
            BranchSimplifier()
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

        target_level = subconfig['target_level']
        while isinstance(cfile.body[0], CppInclude):
            cfile.body.pop(0)
        kernel = cfile.find(FunctionDecl)
        kernel.set_kernel()
        kernel.return_type = None
        kernel.defn[-1] = Assign(ArrayRef(SymbolRef("final"), Constant(0)), kernel.defn[-1].value)
        kernel.params.append(SymbolRef("final", ctypes.POINTER(ctypes.c_double)()))
        for param in kernel.params:
            param.set_global()
        ocl_file = OclFileWrapper("%s_kernel" % kernel.name).visit(cfile)
        ocl_file = DeclarationFiller().visit(ocl_file)
        global_size = reduce(operator.mul, target_level.interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)
        # control = generate_rebuild_control("%s_control" % kernel.name, global_size, local_size, kernel.params, [kernel])
        kernel.name = "%s_kernel" % kernel.name
        # return [control, ocl_file]
        return [ocl_file]


    def finalize(self, transform_result, program_config):

        subconfig, tuner = program_config
        target_level = subconfig['target_level']
        global_size = reduce(operator.mul, target_level.interior_space, 1)
        local_size = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], global_size)

        if (1,) not in target_level.reducer_meshes:
            target_level.reducer_meshes[(1,)] = Mesh((1,))
        final_mesh = target_level.reducer_meshes[(1,)]

        project = Project(transform_result)
        # control = transform_result[0]
        kernel = transform_result[0]

        name = kernel.name
        kernel = cl.clCreateProgramWithSource(target_level.context, kernel.codegen()).build()[name]
        kernel.argtypes = (cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem)
        kernel = KernelRunManager(kernel, global_size, local_size)

        # typesig = [ctypes.c_int, cl.cl_command_queue, cl.cl_kernel, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem]
        fn = RebuildOclFunction()
        # fn = fn.finalize(control.name, project, ctypes.CFUNCTYPE(*typesig),
        #                  target_level, [kernel], (final_mesh,))
        fn = fn.finalize(project, target_level, [kernel], (final_mesh,))
        return fn


def generate_rebuild_control(name, global_size, local_size, kernel_params, kernels):
    defn = []
    defn.append(ArrayDef(SymbolRef("global", ctypes.c_ulong()), 1, Array(body=[Constant(global_size)])))
    defn.append(ArrayDef(SymbolRef("local", ctypes.c_ulong()), 1, Array(body=[Constant(local_size)])))
    # defn.append(Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0)))
    for kernel in kernels:
        kernel_name = kernel.find(FunctionDecl, kernel=True).name
        for param, num in zip(kernel_params, range(len(kernel_params))):
            if isinstance(param, ctypes.POINTER(ctypes.c_double)):
                set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                     [SymbolRef(kernel_name),
                                      Constant(num),
                                      FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
                                      Ref(SymbolRef(param.name))])
            else:
                set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                     [SymbolRef(kernel_name),
                                      Constant(num),
                                      Constant(ctypes.sizeof(param.type)),
                                      Ref(SymbolRef(param.name))])
            defn.append(set_arg)
        enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
            SymbolRef("queue"),
            SymbolRef(kernel_name),
            Constant(1),
            NULL(),
            SymbolRef("global"),
            SymbolRef("local"),
            Constant(0),
            NULL(),
            NULL()
        ])
        defn.append(enqueue_call)
    defn.append(ArrayRef(SymbolRef("return_value", ctypes.c_double()), Constant(1)))
    defn.append(StringTemplate("""clFinish(queue);"""))
    defn.append(FunctionCall(SymbolRef("clEnqueueReadBuffer"),
                             [
                                 SymbolRef("queue"),
                                 SymbolRef("final"),
                                 SymbolRef("CL_TRUE"),
                                 Constant(0),
                                 Constant(8),
                                 Ref(SymbolRef("return_value")),
                                 Constant(0),
                                 NULL(),
                                 NULL()
                             ]))
    defn.append(Return(ArrayRef(SymbolRef("return_value"), Constant(0))))
    params=[]
    params.append(SymbolRef("queue", cl.cl_command_queue()))
    for kernel in kernels:
        params.append(SymbolRef(kernel.find(FunctionDecl, kernel=True).name, cl.cl_kernel()))
    for param in kernel_params:
        if isinstance(param.type, ctypes.POINTER(ctypes.c_double)):
            params.append(SymbolRef(param.name, cl.cl_mem()))
        else:
            params.append(param)
    func = FunctionDecl(ctypes.c_int(), name, params, defn)
    ocl_include = StringTemplate("""
            #include <stdio.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
    body = [ocl_include, func]
    file = CFile(name=name, body=body, config_target='opencl')
    return file
