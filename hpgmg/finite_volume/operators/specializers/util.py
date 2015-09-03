from __future__ import print_function, division
import ast
import atexit
import copy
import functools
import inspect
import time

from ctree import get_ast
import ctree
from ctree.c.macros import NULL
from ctree.cpp.nodes import CppDefine
from ctree.frontend import dump
from ctree.c.nodes import SymbolRef, MultiNode, Constant, Add, Mul, FunctionCall, Div, Mod, Ref, FunctionDecl, ArrayDef, \
    Assign, Return, Array, CFile, Sub, Deref, Literal
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
import itertools
import operator
import sympy
import pycl as cl
import numpy as np
import ctypes

from hpgmg import finite_volume
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, \
    CompReductionTransformer, AttributeFiller
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexTransformer, \
    IndexOpTransformer, IndexDirectTransformer, AttributeRenamer, LookupSimplificationTransformer, get_name


__author__ = 'nzhang-dev'


def specialized_func_dispatcher(specializers):
    def decorator(func):
        func.specializer = None
        func.is_specialized = False
        func.callable = None
        func.signature = inspect.getargspec(func)
        #print(func)

        def set_specializer():
            if finite_volume.CONFIG.backend not in specializers:
                func.specializer = lambda x: func
                #func.callable = func
            else:
                func.specializer = specializers[finite_volume.CONFIG.backend]
                func.is_specialized = True

        func.set_specializer = set_specializer

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not func.is_specialized and func.specializer is None:
                set_specializer()
                func.callable = func.specializer(get_ast(func))
            return func.callable(*args, **kwargs)
        wrapper.func = func
        return wrapper
    return decorator


def to_macro_function(f, namespace=None, rename=None, index_map=None):
    index_map = index_map or {'index': 'encode'}
    namespace = {} if namespace is None else namespace.copy()
    rename = {} if rename is None else rename.copy()
    tree = get_ast(f).body[0]
    tree = ParamStripper(('self',)).visit(tree)
    name = SymbolRef(tree.name)
    params = [SymbolRef(n.id) for n in tree.args.args]
    self = f.im_self
    namespace.update({'self': self})

    layers = [
        ParamStripper(('self',)),
        AttributeRenamer(rename),
        GeneratorTransformer(namespace),
        CompReductionTransformer(),
        AttributeFiller(namespace),
        IndexTransformer(('index',)),
        LookupSimplificationTransformer(),
        IndexOpTransformer(self.solver.dimensions, index_map),
        IndexDirectTransformer(self.solver.dimensions),
        PyBasicConversions()
    ]
    body = MultiNode(body=[apply_all_layers(layers, node) for node in tree.body]).body[0].value
    define = CppDefine(name='apply_op', params=params, body=body)
    return define


def apply_all_layers(layers, node):
    for layer in layers:
        node = layer.visit(node)
    return node


class LayerPrinter(ast.NodeVisitor):
    def visit(self, node):
        print(dump(node))
        return node


def validateCNode(node):
    for node in ast.walk(node):
        if not isinstance(node, ctree.c.nodes.CNode):
            print('FAILED:')
            print(dump(node))
            return False
    return True


def include_mover(node):
    includes = set()
    defines = set()

    class includeFinder(ast.NodeTransformer):
        def visit_CppInclude(self, node):
            includes.add(node)
        def visit_CppDefine(self, node):
            defines.add(node)

    includeFinder().visit(node)
    node.body = [
        include for include in includes
    ]+[
        define for define in defines
    ]+node.body
    return node


def time_this(func):
    time_this.names.append(func.__name__)
    timings = []

    def wrapper(*args, **kwargs):
        a = time.time()
        res = func(*args, **kwargs)
        timings.append(time.time() - a)
        return res
    wrapper.total_time = 0

    @atexit.register
    def dump_time():
        if finite_volume.CONFIG and finite_volume.CONFIG.verbose and timings:
            max_len = max(len(i) for i in time_this.names)
            format_statement = "Function: {"
            print('Function: {name:{width}.{width}}  Total time: {time:15.10f}   calls: {calls:10d}'.format(
                name=func.__name__, width=max_len, time=sum(timings), calls=len(timings)))
    return wrapper
time_this.names = []


def profile(func):
    print("profiling with line_profiler")
    if 'profile' in __builtins__:
        return __builtins__['profile'](func)
    return func


def sympy_to_c(exp, sym_name='x'):
    if isinstance(exp, sympy.Number):
        if exp.is_Float:
            return Constant(float(exp))
        if exp.is_Integer:
            return Constant(int(exp))

    args = [sympy_to_c(i, sym_name) for i in exp.args]

    if isinstance(exp, sympy.Add):
        return functools.reduce(Add, args)

    if isinstance(exp, sympy.Mul):
        return functools.reduce(Mul, args)

    if isinstance(exp, sympy.Pow):
        return FunctionCall(SymbolRef("pow"), args)

    if isinstance(exp,
                  (
                      sympy.functions.elementary.trigonometric.TrigonometricFunction,
                      sympy.functions.elementary.hyperbolic.HyperbolicFunction
                  )):
        return FunctionCall(SymbolRef(type(exp).__name__), args)

    if isinstance(exp, sympy.Symbol):
        s = sym_name + str(exp)[1:]
        return SymbolRef(s)

    raise ValueError("Could not parse {}".format(exp))


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.dependencies = set()
        self.defines = set()
        self.define = False
        super(Analyzer, self).__init__()

    def visit_Assign(self, node):
        self.define = True
        for target in node.targets:
            self.visit(target)
        self.define = False
        node = self.generic_visit(node.value)

    def visit_Subscript(self, node):
        state, self.define = self.define, False
        self.generic_visit(node)
        self.define = state

    def visit_Name(self, node):
        if self.define:
            self.defines.add(node.id)
        elif node.id not in self.defines:
            self.dependencies.add(node.id)

    def visit_Attribute(self, node):
        name = get_name(node)
        levels = name.split('.')
        parts = []
        for level in levels:
            parts.append(level)
            s = ".".join(parts)
            if s in self.defines | self.dependencies:
                return
        if self.define:
            self.defines.add(name)
        elif name not in self.defines:
            self.dependencies.add(name)

    def visit_For(self, node):
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                self.defines.add(t.id)
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def __str__(self):
        return "{}\t{}".format(self.defines, self.dependencies)


def analyze_dependencies(tree):
    #print(dump(tree))

    analyzer = Analyzer()
    analyzer.visit(tree)
    return analyzer


def find_fusible_blocks(tree, namespace):
    """
    :param tree: Tree you're searching for fusible blocks
    :param specialized_functions: functions that have been specialized
    :return: List of blocks that can be specialized
    """
    is_fusible(tree, namespace)

    class BlockTransformer(ast.NodeTransformer):
        def __init__(self, names):
            self.names = names
        def visit_For(self, node):

            #TODO: Need a heuristic for fusion. Can't just fuse random things

            ns_copy = self.names.copy()
            if node.fusible:
                #try to fuse the entire thing in
                try:
                    print("Fusing")
                    return node
                except AttributeError:
                    pass  # failed
            # try to find a chunk that is fusible

            groups = itertools.groupby(node.body, operator.attrgetter('fusible'))


def is_fusible(tree, namespace):
    operations = tuple(PyBasicConversions.PY_OP_TO_CTREE_OP.keys())

    def has_object_in_namespace(s, namespace):
        split = s.split('.')
        obj_name = split.pop(0)
        if obj_name not in namespace:
            return False
        obj = namespace[obj_name]
        for name in split:
            if not hasattr(obj, name):
                return False
            obj = getattr(obj, name)
        return True

    class FusionFinder(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            node.fusible = all(child.fusible for child in ast.iter_child_nodes(node))
            return node

        def visit(self, node):
            node = super(FusionFinder, self).visit(node)
            if hasattr(node, 'fusible'):
                pass
            elif hasattr(PyBasicConversions, 'visit_' + type(node).__name__) or isinstance(node, operations):
                node.fusible = all(child.fusible for child in ast.iter_child_nodes(node))
            else:
                node.fusible = False
            return node

        def visit_Attribute(self, node):
            node.fusible = has_object_in_namespace(get_name(node), namespace)
            return node

        def visit_Name(self, node):
            return self.visit_Attribute(node)

    return FusionFinder().visit(tree).fusible


def get_object(name, namespace, allow_builtins=False):
    split = name.split(".")
    obj_name = split.pop(0)
    if allow_builtins:
        namespace.update(__builtins__)
    if obj_name not in namespace:
        raise NameError('name {} is not defined'.format(obj_name))
    obj = namespace[obj_name]
    while split:
        attr_name = split.pop(0)
        if not hasattr(obj, attr_name):
            raise AttributeError('{} object has no attribute {}'.format(type(obj), attr_name))
        obj = getattr(obj, attr_name)
    return obj


def string_to_ast(s):
    """
    converts string s to nested ast.Attribute and Name nodes
    """
    s_list = s.split(".")
    result = ast.Name(id=s_list.pop(0), ctx=ast.Load())
    for attrib in s_list:
        result = ast.Attribute(value=result, attr=attrib, ctx=ast.Load())
    return result


def get_arg_spec(f):
    if hasattr(f, 'argspec'):
        return f.argspec[:]
    f.argspec = inspect.getargspec(f).args
    return f.argspec[:]

def compute_local_work_size(device, shape):
    ndim = len(shape)
    interior_space = reduce(operator.mul, shape, 1)
    local_cube_dim, local_size = 1, 1
    max_size = min(device.max_work_group_size, interior_space)

    while local_cube_dim ** ndim < max_size:
        local_cube_dim += 1
        if interior_space % (local_cube_dim ** ndim) == 0:
            local_size = local_cube_dim ** ndim
    return local_size

def compute_largest_local_work_size(device, global_size):
    local_size = min(device.max_work_group_size, global_size)
    while global_size % local_size != 0:
        local_size -= 1
    return local_size

def flattened_to_multi_index(flattened_id_symbol, shape, multipliers=None, offsets=None):

    # flattened_id should be a node
    # offsets applied after multipliers

    body = []
    ndim = len(shape)
    for i in range(ndim):
        mod_size = reduce(operator.mul, shape[i:], 1)
        div_size = reduce(operator.mul, shape[(i + 1):], 1)
        stmt = Div(Mod(flattened_id_symbol, Constant(mod_size)), Constant(div_size))
        if multipliers:
            stmt = Mul(stmt, Constant(multipliers[i]))
        if offsets:
            stmt = Add(stmt, Constant(offsets[i]))
        body.append(stmt)
    return body

def new_generate_control(name, global_size, local_size, kernel_params, kernels, other=None):
    # assumes that all kernels take the same arguments and that they all use the same global and local size!
    defn = []
    defn.append(StringTemplate("""clock_t start, diff;"""))
    defn.append(ArrayDef(SymbolRef("global", ctypes.c_ulong()), 1, Array(body=[Constant(global_size)])))
    defn.append(ArrayDef(SymbolRef("local", ctypes.c_ulong()), 1, Array(body=[Constant(local_size)])))
    defn.append(Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0)))
    defn.append(StringTemplate("""start = clock();"""))
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
    defn.append(StringTemplate("""clFinish(queue);"""))
    defn.append(StringTemplate("""diff = clock() - start;"""))
    defn.append(StringTemplate("""total_time[0] = total_time[0] + (diff / (float) CLOCKS_PER_SEC);"""))
    defn.append(Return(SymbolRef("error_code")))
    params=[]
    params.append(SymbolRef("queue", cl.cl_command_queue()))
    for kernel in kernels:
        params.append(SymbolRef(kernel.find(FunctionDecl, kernel=True).name, cl.cl_kernel()))
    for param in kernel_params:
        if isinstance(param.type, ctypes.POINTER(ctypes.c_double)):
            params.append(SymbolRef(param.name, cl.cl_mem()))
        else:
            params.append(param)
    params.append(SymbolRef("total_time", ctypes.POINTER(ctypes.c_float)()))
    func = FunctionDecl(ctypes.c_int(), name, params, defn)
    ocl_include = StringTemplate("""
            #include <stdio.h>
            #include <time.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
    body = [ocl_include, func]
    if other:
        body.extend(other)
    file = CFile(name=name, body=body, config_target='opencl')
    return file