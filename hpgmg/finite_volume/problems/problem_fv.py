from __future__ import print_function

import numpy as np
import sympy

from hpgmg.finite_volume.hpgmg_exception import HpgmgException
from hpgmg.finite_volume.operators.specializers.util import time_this, profile
from hpgmg.finite_volume.problems.algebraic_problem import SymmetricAlgebraicProblem
from hpgmg.tools.text_indenter import TextIndenter


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ProblemFV(SymmetricAlgebraicProblem):
    def __init__(self, dimensions=3, add_4th_order_correction=False, cell_size=None):
        if add_4th_order_correction and cell_size is None:
            raise HpgmgException("ProblemFv requires cells size")

        period = 2.0 * np.pi
        power = 7.0
        expr = "sin({period}*x) ** {power}".format(period=period, power=power)
        self.cell_size = cell_size

        # this computes the base term of the n dimensional
        super(ProblemFV, self).__init__(expr, dimensions)

        if add_4th_order_correction:
            self.second_derivatives = []
            # requires two loops so original expression is available for derivative
            for current_dim in range(dimensions):
                self.second_derivatives.append(self.get_derivative(current_dim, derivative=2))

            for current_dim in range(dimensions):
                self.expression = self.expression + sympy.sympify("{}*{}/24.0*{}".format(
                    cell_size, cell_size, self.second_derivatives[current_dim]))

        # print("ProblemFV, expression {}".format(self.expression))

        beta_expression = self.expand_dimensions("sin({}*x)".format(period))
        beta_expression *= 0.25
        beta_expression += 1.0

        if add_4th_order_correction:
            self.beta_second_derivatives = []
            # requires two loops so original expression is available for derivative
            for current_dim in range(dimensions):
                self.beta_second_derivatives.append(beta_expression.diff(self.symbols[current_dim], derivative=2))

            for current_dim in range(dimensions):
                beta_expression = self.expression + sympy.sympify("{}*{}/24.0*{}".format(
                    cell_size, cell_size, self.beta_second_derivatives[current_dim]))

        self.beta_expression = beta_expression

    @time_this
    @profile
    def initialize_problem(self, solver, level):
        if level.cell_size != self.cell_size:
            raise HpgmgException("Initialize ProblemFV cell size of level {} does not match problem {}".format(
                level.cell_size, self.cell_size
            ))

        self.initialize_mesh(level, level.right_hand_side, self.expression, level.coord_to_cell_center_point)  # , dump=True)
        level.exact_solution.fill(0.0)

        level.alpha.fill(1.0)

        for dim in range(self.dimensions):
            if solver.is_variable_coefficient:
                self.initialize_face_mesh(level, level.beta_face_values[dim],
                                          solver.beta_generator.get_beta_fv_expression(add_4th_order_correction=True,
                                                                                       cell_size=level.cell_size), dim)
            else:
                level.beta_face_values[dim].fill(1.0)

    def initialize_problem_codegen(self, solver, level):
        """
        TODO: This is a work in progress, the goal is to refactor problem initialization in python
        to look more like Sam's method where a single loop initializes all the meshes.

        :param solver:
        :param level:
        :return:
        """
        text = TextIndenter()
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
        text += "level.exact_solution[index] = 1.0"
        text += "level.right_hand_side[index] = {}".format(self.expression)

        text += [
            "z{dim} = x{dim} - {size}".format(
                dim=dim, size=level.cell_size/2.0
            )
            for dim in range(solver.dimensions)
        ]

        for dimension in range(solver.dimensions):
            text += "level.beta_face_values[{dim}][index] = {beta_expr}".format(
                    dim=dimension,
                    beta_expr=solver.beta_generator.get_beta_fv_expression(add_4th_order_correction=True,
                                                                           cell_size=level.cell_size,
                                                                           face_index=dimension,
                                                                           alternate_var_name="z")
            )

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
    print("I am in deco")
    return f