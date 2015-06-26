from __future__ import print_function, division
import ast
import atexit
import functools
import time

from ctree import get_ast
import ctree
from ctree.cpp.nodes import CppDefine
from ctree.frontend import dump
from ctree.c.nodes import SymbolRef, MultiNode, Constant, Add, Mul, FunctionCall
from ctree.transformations import PyBasicConversions
import sympy

from hpgmg import finite_volume
from hpgmg.finite_volume.operators.transformers.generator_transformers import GeneratorTransformer, CompReductionTransformer, \
    AttributeFiller
from hpgmg.finite_volume.operators.transformers.utility_transformers import ParamStripper, IndexTransformer, \
    IndexOpTransformer, IndexDirectTransformer, AttributeRenamer, LookupSimplificationTransformer


__author__ = 'nzhang-dev'

def specialized_func_dispatcher(specializers):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if finite_volume.CONFIG.backend not in specializers:
                specializer = lambda x: func
            else:
                specializer = specializers[finite_volume.CONFIG.backend]
            callable_thing = specializer(get_ast(func))
            return callable_thing(*args, **kwargs)
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
            maxlen = max(len(i) for i in time_this.names)
            print('Function:', func.__name__.ljust(maxlen), 'Total time:', sum(timings), 'calls:', len(timings), sep="\t")
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

    if isinstance(exp, sympy.functions.elementary.trigonometric.TrigonometricFunction):
        return FunctionCall(SymbolRef(type(exp).__name__), args)

    if isinstance(exp, sympy.Symbol):
        s = sym_name + str(exp)[1:]
        return SymbolRef(s)

    raise ValueError("Could not parse {}".format(exp))