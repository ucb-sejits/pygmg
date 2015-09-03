import sympy
import operator
import functools
import numpy as np
from hpgmg.finite_volume.operators.specializers.initialize_mesh_specializer import CInitializeMesh

from hpgmg.finite_volume.operators.specializers.util import time_this, profile, specialized_func_dispatcher
from hpgmg.finite_volume.problems.problem import Problem

__author__ = 'nzhang-dev'


class AlgebraicProblem(Problem):
    pass


class SymmetricAlgebraicProblem(AlgebraicProblem):
    def __init__(self, expression, dimensions, reduction_operator=operator.mul):
        self.dimensions = dimensions
        self.symbols = [sympy.Symbol("x{}".format(d)) for d in range(self.dimensions)]

        expression = sympy.sympify(expression)
        symbols = expression.free_symbols
        if len(symbols) != 1:
            raise ValueError("Too many free variables in expression")
        symbol = symbols.pop()
        self.expression = functools.reduce(
            reduction_operator,
            (expression.xreplace({symbol: sympy.Symbol("x{}".format(i))}) for i in range(dimensions))
        )
        #print(expression)

    def expand_dimensions(self, expression, reduction_operator=operator.mul):
        expression = sympy.sympify(expression)
        symbols = expression.free_symbols
        if len(symbols) != 1:
            raise ValueError("Too many free variables in expression")
        symbol = symbols.pop()
        expanded_expression = functools.reduce(
            reduction_operator,
            (expression.xreplace({symbol: sympy.Symbol("x{}".format(i))}) for i in range(self.dimensions))
        )
        return expanded_expression

    def get_derivative(self, dim, derivative=0):
        #print(self.expression.diff(self.symbols[dim-1], derivative))
        return self.expression.diff(self.symbols[dim-1], derivative)

    @staticmethod
    def get_func(func, symbols=None):
        symbols = symbols or sorted(list(func.free_symbols), key=str)
        lambda_func = sympy.lambdify(symbols, func, "numpy")
        numpy_func = np.frompyfunc(lambda_func, len(symbols), 1)
        return numpy_func

    @specialized_func_dispatcher({
        'c': CInitializeMesh,
        'omp': CInitializeMesh
    })
    def initialize_mesh(self, level, mesh, exp, dump=False):
        func = self.get_func(exp, self.symbols)
        # print("expression {}".format(exp))
        for coord in level.interior_points():
            if dump:
                x, y, z = level.coord_to_cell_center_point(coord)
                f = func(*level.coord_to_cell_center_point(coord))
                print("Coordinate ({:12.10f},{:12.10f},{:12.10f}) -> {:10.6g}".format(x, y, z, f))
            mesh[coord] = func(*level.coord_to_cell_center_point(coord))

    @specialized_func_dispatcher({
        'c': CInitializeMesh,
        'omp': CInitializeMesh
    })
    def initialize_face_mesh(self, level, mesh, exp, dimension, dump=False):
        func = self.get_func(exp, self.symbols)
        # print("expression {}".format(exp))
        for coord in level.indices():
            # if dump:
            #     x, y, z = level.coord_to_face_center_point(coord, dimension)
            #     f = func(*level.coord_to_face_center_point(coord, dimension))
            #     print("Coordinate ({:12.10f},{:12.10f},{:12.10f}) -> {:10.6g}".format(x, y, z, f))
            mesh[coord] = func(*level.coord_to_face_center_point(coord, dimension))

    @time_this
    @profile
    def initialize_problem(self, solver, level):
        solver.initialize_mesh(level, level.exact_solution, self.expression)
        level.exact_solution.fill(0.0)

        level.alpha.fill(1.0)

        for dim in range(self.dimensions):
            if self.use_variable_coefficient:
                self.initialize_face_mesh(level, level.beta_face_values[dim], self.beta_expression, dim)
            else:
                level.beta_face_values[dim].fill(1.0)
