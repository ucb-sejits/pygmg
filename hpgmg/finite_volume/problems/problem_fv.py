from __future__ import print_function
import numpy as np

from hpgmg.finite_volume.problems.algebraic_problem import SymmetricAlgebraicProblem

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ProblemFV(SymmetricAlgebraicProblem):
    def __init__(self, dimensions=3, shift=0.0):
        period = 2.0 * np.pi
        power = 7.0
        expr = "sin({period}*x) ** {power}".format(period=period, power=power)
        super(ProblemFV, self).__init__(expr, dimensions)

