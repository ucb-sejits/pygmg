import sympy
from hpgmg.finite_volume.problems.problem import Problem
import operator
import functools
import numpy as np

__author__ = 'nzhang-dev'


class AlgebraicProblem(Problem):
    pass

class SymmetricAlgebraicProblem(AlgebraicProblem):
    def __init__(self, expression, dimensions, reduction_operator=operator.mul):
        expression = sympy.sympify(expression)
        symbols = expression.free_symbols
        if len(symbols) != 1:
            raise ValueError("Too many free variables in expression")
        symbol = symbols.pop()
        self.expression = functools.reduce(
            reduction_operator,
            (expression.xreplace({symbol: sympy.Symbol("x{}".format(i))}) for i in range(dimensions))
        )
        self.dimensions = dimensions
        self.symbols = [sympy.Symbol("x{}".format(d)) for d in range(self.dimensions)]
        #print(expression)

    def get_derivative(self, dim, derivative=0):
        #print(self.expression.diff(self.symbols[dim-1], derivative))
        return self.expression.diff(self.symbols[dim-1], derivative)

    @staticmethod
    def get_func(func, symbols=None):
        symbols = symbols or sorted(list(func.free_symbols), key=str)
        lambda_func = sympy.lambdify(symbols, func, "numpy")
        numpy_func = np.frompyfunc(lambda_func, len(symbols), 1)
        return numpy_func
