import __builtin__
import collections
import inspect
import types
from ctree.cpp.nodes import CppInclude
import numpy as np
from ctree.frontend import get_ast, dump
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.transformations import PyBasicConversions
from ctree.c.nodes import FunctionDecl, SymbolRef, CFile
from ctree.nodes import Project
import ctypes
import ast
from hpgmg import finite_volume
from hpgmg.finite_volume.operators.specializers.util import analyze_dependencies, get_object, include_mover, \
    string_to_ast, apply_all_layers
from hpgmg.finite_volume.operators.transformers.utility_transformers import get_name, AttributeGetter, \
    GeneralAttributeRenamer

__author__ = 'nzhang-dev'

class QuickFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.entry_point_name = entry_point_name
        return self

    def __call__(self, *args):
        flattened = [arg.ravel() if isinstance(arg, np.ndarray) else arg for arg in args[:-2]]
        self._c_function(*flattened)

class QuickJit(LazySpecializedFunction):
    @staticmethod
    def parse_param_type(param):
        if isinstance(param, int):
            return ctypes.c_int()
        elif isinstance(param, float):
            return ctypes.c_double()
        elif isinstance(param, np.ndarray):
            t = param.dtype.type
            out = ctypes.c_void_p
            if issubclass(t, np.int64):
                out = ctypes.c_longlong
            elif issubclass(t, np.int):
                out = ctypes.c_int
            elif issubclass(t, np.double):
                out = ctypes.c_double
            elif issubclass(t, np.float):
                out = ctypes.c_float
            return ctypes.POINTER(out)()

    @staticmethod
    def parse_cfunc_type(param):
        if isinstance(param, int):
            return ctypes.c_int
        elif isinstance(param, float):
            return ctypes.c_double
        elif isinstance(param, np.ndarray):
            return np.ctypeslib.ndpointer(param.dtype, 1, param.size)

    class Subconfig(list):
        def __hash__(self):
            h = 0
            for i in self:
                if isinstance(i, np.ndarray):
                    h ^= hash(str(i.dtype))
                else:
                    h ^= hash(type(i))
            return h

    def args_to_subconfig(self, args):
        #print(args)
        parts = [arg for arg in args if isinstance(arg, (np.ndarray, int, float))]
        parts.extend(args[-2:])
        return self.Subconfig(parts)

    def transform(self, tree, program_config):
        subconfig, tuner_config = program_config
        subconfig, local_namespace, global_namespace = subconfig[:-2], subconfig[-2], subconfig[-1]
        #print set(local_namespace)
        dependencies = analyze_dependencies(tree)
        #tree = PyBasicConversions().visit(tree)
        #print(tree)
        #print(dependencies)
        deps = dependencies.dependencies.copy()
        pre_defined_functions = {'range'}
        deps -= pre_defined_functions
        deps = set(dep.split(".")[0] for dep in deps)

        full_namespace = global_namespace.copy()
        full_namespace.update(local_namespace)
        call_nodes = {get_name(node.func): node for node in ast.walk(tree) if isinstance(node, ast.Call)}
        files = []
        print("IN TRANSFORM")
        for func_name, node in call_nodes.items():
            func_obj = get_object(func_name, full_namespace, allow_builtins=True)
            if func_obj.__name__ in dir(__builtin__):
                continue
            if getattr(func_obj, 'func', None):
                # specialized func
                func_obj = func_obj.func
                func_obj.set_specializer()
                arguments = []
                specializer = func_obj.specializer
                print(specializer.arg_spec, node.args)
                for call_arg in node.args:
                    name = get_name(call_arg)
                    arguments.append(get_object(name, full_namespace))
                sub_program_config = specializer.get_program_config(args=arguments)
                #dir_name = specializer.config_to_dirname(sub_program_config)
                files.extend(specializer.run_transform(sub_program_config))


        #print(deps)
        code_hash = hash(ast.dump(self.tree, True, True))
        #print(subconfig)
        param_types = [self.parse_param_type(arg) for arg in subconfig]
        param_names = sorted(deps)

        layers = [
            GeneralAttributeRenamer(lambda x: x.split(".")[-1])
            #AttributeGetter(full_namespace)
        ]

        tree = apply_all_layers(layers, tree)
        print(dump(tree))
        raise Exception()
        params = [SymbolRef(name=param_name, sym_type=param_type) for param_name, param_type in zip(param_names, param_types)]
        function = FunctionDecl(name=SymbolRef('on_the_fly_{}'.format(abs(code_hash))),
                                params=params,
                                defn=[tree]
                                )
        #print(function)
        func_file = CFile(body=[function])
        for file_id, f in enumerate(files):
            f.name = "f_{}".format(file_id)
        func_file.body.extend(
            CppInclude(f.name, False) for f in files
        )
        func_file = include_mover(func_file)
        all_files = [func_file] + files
        for f in all_files:
            print(f)
        return all_files

    def finalize(self, transform_result, program_config):
        #print(transform_result[0])
        subconfig, tuner_config = program_config
        subconfig = subconfig[:-2]
        code_hash = hash(ast.dump(self.tree, True, True))
        name = 'on_the_fly_{}'.format(abs(code_hash))
        fn = QuickFunction()
        param_types = [self.parse_cfunc_type(arg) for arg in subconfig]
        print(param_types)
        func_type = ctypes.CFUNCTYPE(None, *param_types)
        return fn.finalize(name, Project(transform_result), func_type)

class ConvertToCall(ast.NodeTransformer):
    def __init__(self, function_name, is_instancemethod=False):
        self.counter = 0
        self.function_name = function_name
        self.specialized = []
        self.is_instancemethod = is_instancemethod

    def visit_For(self, node):
        spec_name = '_func_{}'.format(self.counter)
        if self.is_instancemethod:
            func_name = ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()), attr=self.function_name,
                ctx=ast.Load()
            )
        else:
            func_name = ast.Name(id=self.function_name, ctx=ast.Load())

        func_name = ast.Attribute(value=func_name, attr=spec_name, ctx=ast.Load())
        self.counter += 1
        self.specialized.append(node)
        analysis = analyze_dependencies(node)
        dependencies = sorted(analysis.dependencies - set(__builtin__.__dict__))
        args = [string_to_ast(dep) for dep in dependencies]
        call = lambda name: ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[], keywords=[], starargs=None, kwargs=None)
        args.append(call('locals'))
        args.append(call('globals'))
        exp = ast.Expr(value=ast.Call(func=func_name, args=args, keywords=[], starargs=None, kwargs=None))
        return exp

def partial_jit(function):

    def wrapper(*args, **kwargs):
        #print str(finite_volume.CONFIG)
        if not hasattr(function, 'callable') and finite_volume.CONFIG and finite_volume.CONFIG.backend != 'python':
            #print("JIT")
            tree = get_ast(function)
            spec = inspect.getargspec(function)
            is_instancemethod = len(spec) > 0 and spec.args[0] == 'self'
            c2c = ConvertToCall(function.__name__, is_instancemethod)
            #print(type(function), isinstance(function, types.MethodType))
            tree = c2c.visit(tree)
            jits = [QuickJit(py_ast=sub_tree) for sub_tree in c2c.specialized]
            #print(dump(c2c.specialized[0]))
            print(dump(tree))
            #print(len(jits))
            f_tree = tree.body[0]
            f_tree.decorator_list = [i for i in f_tree.decorator_list if not isinstance(i, ast.Name) and i.id == 'jit_this_crap']
            #print(dump(tree))
            ast.fix_missing_locations(tree)
            #print(dump(tree))
            code = compile(tree, '<string>', 'exec')
            exec code in globals(), locals()
            function_obj = locals()[function.__name__]
            for i, jit_func in enumerate(jits):
                setattr(wrapper, '_func_{}'.format(i), jit_func)
            function.callable = function_obj
            #print(function.callable)
        else:
            function.callable = function
        return function.callable(*args, **kwargs)
    return wrapper