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
            # for current_dim in range(dimensions):
            #     second_derivative = self.get_derivative(current_dim, derivative=2)
            #     self.expression = sympy.sympify(self.expression.__repr__() + " + {}*{}/24.0*{}".format(
            #         cell_size, cell_size, second_derivative))
            pass

        print("ProblemFV, expression {}".format(self.expression))

