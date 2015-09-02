from __future__ import print_function
import numpy as np
import sympy
from hpgmg.finite_volume.hpgmg_exception import HpgmgException
from hpgmg.finite_volume.operators.specializers.util import time_this, profile

from hpgmg.finite_volume.problems.algebraic_problem import SymmetricAlgebraicProblem

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

    @time_this
    @profile
    def initialize_problem(self, solver, level):
        if level.cell_size != self.cell_size:
            raise HpgmgException("Initialize ProblemFV cell size of level {} does not match problem {}".format(
                level.cell_size, self.cell_size
            ))

        solver.initialize_mesh(level, level.right_hand_side, self.expression)
