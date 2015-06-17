import ctypes
from ctree.c.nodes import FunctionCall, SymbolRef, Add, Mul, Constant

__author__ = 'nzhang-dev'

import sympy

def register_setup(func):
    def decorator(setup):
        setup(func)
    return decorator


def sympy_exp_to_c(expr):
    exp_type = type(expr)
    if exp_type in sympy_exp_to_c.op_map:
        node_type = sympy_exp_to_c.op_map[exp_type]
        children = [sympy_exp_to_c(child) for child in expr.args]
        return node_type(*children)
    elif exp_type in sympy_exp_to_c.numbers:
        return sympy_exp_to_c.numbers[exp_type](expr)
    raise ValueError("Cannot convert {} to a C AST Node".format(expr))


@register_setup(sympy_exp_to_c)
def _setup_sympy_exp_to_c(func):
    func.suffix_map = {
        ctypes.c_double: '',
        ctypes.c_float: 'f',
        ctypes.c_longdouble: 'l'
    }
    func.op_map = {
        sympy.Pow: lambda x, y: FunctionCall(SymbolRef('pow'), args=[x, y]),
        sympy.Add: Add,
        sympy.Mul: Mul,
    }
    trig_funcs = [
        sympy.sin, sympy.cos, sympy.tan,
        sympy.asin, sympy.acos, sympy.atan,
        sympy.sinh, sympy.cosh, sympy.tanh,
        sympy.asinh, sympy.acosh, sympy.atanh
    ]
    trig_functions = {}
    for trig_func in trig_funcs:
        def create_lambda(trig_func):
            return lambda x: FunctionCall(SymbolRef(trig_func.__name__), args=[x])
        trig_functions[trig_func] = create_lambda(trig_func)
    func.op_map.update(trig_functions)

    func.numbers = {
        sympy.Integer: lambda x: Constant(int(x)),
        sympy.Float: lambda x: Constant(float(x)),
        sympy.Symbol: lambda x: SymbolRef(str(x))
    }