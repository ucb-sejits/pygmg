import operator
import functools

import sympy
import numpy as np

from hpgmg.finite_volume.operators.specializers.initialize_mesh_specializer import CInitializeMesh
from hpgmg.finite_volume.operators.specializers.util import time_this, profile, specialized_func_dispatcher
from hpgmg.finite_volume.problems.problem import Problem
from hpgmg.tools.text_indenter import TextIndenter


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
        return self.expression.diff(self.symbols[dim], derivative)

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
        print("expression {}".format(exp))
        # for coord in level.indices():
        for coord in level.interior_points():
            if dump and self.dimensions == 3:
                x, y, z = level.coord_to_cell_center_point(coord)
                f = func(*level.coord_to_cell_center_point(coord))
                print("Coordinate ({:12.10f},{:12.10f},{:12.10f}) -> {:10.6g}".format(x, y, z, f))
            mesh[coord] = func(*level.coord_to_cell_center_point(coord))

    @specialized_func_dispatcher({
        'c': CInitializeMesh,
        'omp': CInitializeMesh
    })
    def initialize_face_mesh(self, level, mesh, exp, dimension):
        func = self.get_func(exp, self.symbols)
        # print("expression {}".format(exp))
        for coord in level.indices():
            mesh[coord] = func(*level.coord_to_face_center_point(coord, dimension))

    @time_this
    @profile
    def initialize_problem(self, solver, level):
        alpha = 1.0
        beta = 1.0
        self.initialize_mesh(level, level.exact_solution, self.expression)

        beta_expression = solver.beta_generator.get_beta_expression() if solver.is_variable_coefficient else beta
        beta_first_derivative = [sympy.diff(beta_expression, sym) for sym in self.symbols]
        u_first_derivative = [sympy.diff(self.expression, sym) for sym in self.symbols]
        bu_derivative_1 = sum(a * b for a, b in zip(beta_first_derivative, u_first_derivative))
        u_derivative_2 = [sympy.diff(self.expression, sym, 2) for sym in self.symbols]
        f_exp = solver.a * alpha * self.expression - solver.b * (
            bu_derivative_1 + beta_expression * sum(u_derivative_2))

        self.initialize_mesh(level, level.right_hand_side, f_exp)

        level.alpha.fill(1.0)

        for dim in range(self.dimensions):
            if solver.is_variable_coefficient:
                self.initialize_face_mesh(level, level.beta_face_values[dim], beta_expression, dim)
            else:
                level.beta_face_values[dim].fill(1.0)

    def orig_initialize_problem_codegen(self, solver, level):
        """
        TODO: This is a work in progress, the goal is to refactor problem initialization in python
        to look more like Sam's method where a single loop initializes all the meshes.

        :param solver:
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0

        loop_vars = ", ".join(["i{}".format(dim) for dim in range(solver.dimensions)])
        lines = ["for index in level.indices():"]
        lines += [
            "    {}".format(line)
            for line in level.coord_to_cell_center_formula("x", "index")
        ]
        lines.append("    level.alpha[index] = 1.0".format())
        lines.append("    level.right_hand_side[index] = {}".format(loop_vars, self.expression))

        beta_expression = solver.beta_generator.get_beta_expression() if solver.is_variable_coefficient else beta
        beta_first_derivative = [sympy.diff(beta_expression, sym) for sym in self.symbols]
        u_first_derivative = [sympy.diff(self.expression, sym) for sym in self.symbols]
        bu_derivative_1 = sum(a * b for a, b in zip(beta_first_derivative, u_first_derivative))
        u_derivative_2 = [sympy.diff(self.expression, sym, 2) for sym in self.symbols]
        f_exp = solver.a * alpha * self.expression - solver.b * (
            bu_derivative_1 + beta_expression * sum(u_derivative_2))

        lines.append("    level.exact_solution[index] = {}".format(f_exp))

        lines += [
            "    z{dim} = x{dim} - {size}".format(
                dim=dim, size=level.cell_size/2.0
            )
            for dim in range(solver.dimensions)
        ]

        for dimension in range(solver.dimensions):
            lines.append(
                "    level.beta_face_values[{dim}][({loop_vars})] = {beta_expr}".format(
                    dim=dimension,
                    loop_vars=loop_vars,
                    beta_expr=solver.beta_generator.get_beta_expression(face_index=dimension, alternate_var_name="z")
                )
            )

        for index, line in enumerate(lines):
            print("{:>4d}  {}".format(index, line))

    def initialize_problem_codegen(self, solver, level):
        """
        TODO: This is a work in progress, the goal is to refactor problem initialization in python
        to look more like Sam's method where a single loop initializes all the meshes.

        :param solver:
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0

        text = TextIndenter()
        # text += "@specialized_func_dispatcher({})"
        text += "def init_problem(level):"
        text.indent()
        text += "from math import sin, cos, tanh"
        text += "for index in level.indices():"
        text.indent()
        text += [
            "{}".format(line)
            for line in level.coord_to_cell_center_formula("x", "index")
        ]
        text += "level.alpha[index] = 1.0"
        text += "level.right_hand_side[index] = {}".format(self.expression)

        beta_expression = solver.beta_generator.get_beta_expression() if solver.is_variable_coefficient else beta
        beta_first_derivative = [sympy.diff(beta_expression, sym) for sym in self.symbols]
        u_first_derivative = [sympy.diff(self.expression, sym) for sym in self.symbols]
        bu_derivative_1 = sum(a * b for a, b in zip(beta_first_derivative, u_first_derivative))
        u_derivative_2 = [sympy.diff(self.expression, sym, 2) for sym in self.symbols]
        f_exp = solver.a * alpha * self.expression - solver.b * (
            bu_derivative_1 + beta_expression * sum(u_derivative_2))

        text += "level.exact_solution[index] = {}".format(f_exp)

        text += [
            "z{dim} = x{dim} - {size}".format(
                dim=dim, size=level.cell_size/2.0
            )
            for dim in range(solver.dimensions)
        ]

        for dimension in range(solver.dimensions):
            text += "level.beta_face_values[{dim}][index] = {beta_expr}".format(
                    dim=dimension,
                    beta_expr=solver.beta_generator.get_beta_expression(face_index=dimension, alternate_var_name="z")
                )
        text.outdent()
        text += "level.right_hand_side.dump('RHS', force_dump=True)"
        text += "print('hello world')"
        for index, line in enumerate(text.lines):
            print("{:>4d}  {}".format(index, line))

        exec(text.__str__())

        init_problem(level)

        # import ast
        #
        # import ctree
        #
        # tree = ast.parse(text.__str__())
        # print("dumping tree")
        # ast.dump(tree)
        # ctree.browser_show_ast(tree)


def deco(f):
    print("Here I am in deco")
    return f