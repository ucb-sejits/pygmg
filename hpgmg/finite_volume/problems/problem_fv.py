from __future__ import print_function
import numpy as np
import sympy

from hpgmg.finite_volume.problems.algebraic_problem import SymmetricAlgebraicProblem

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ProblemFV(SymmetricAlgebraicProblem):
    def __init__(self, dimensions=3, cell_size=1/32.0, add_4th_order_correction=False):
        period = 2.0 * np.pi
        power = 7.0
        expr = "sin({period}*x) ** {power}".format(period=period, power=power)

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
