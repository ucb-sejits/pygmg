from __future__ import print_function
from math import tanh, sin, cos

import numpy as np
from hpgmg.finite_volume.constants import Constants

import hpgmg.finite_volume.operators.misc as misc
from hpgmg.finite_volume.operators.problem import Problem
from hpgmg.finite_volume.space import Vector

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ProblemP4(Problem):
    @staticmethod
    def symbolic_version(coord):
        from sympy import diff, Symbol, sin
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        shift = Symbol('shift')

        expr = (
            (1.0 * x**4 - 2.0 * x**3 + 1.0 * x**2 + shift) +
            (1.0 * y**4 - 2.0 * y**3 + 1.0 * y**2 + shift) +
            (1.0 * z**4 - 2.0 * z**3 + 1.0 * z**2 + shift)
        )

        d_dx = diff(expr, x)
        d_dy = diff(expr, y)
        d_dz = diff(expr, z)

        d2_dx2 = diff(expr, x, 2)
        d2_dy2 = diff(expr, y, 2)
        d2_dz2 = diff(expr, z, 2)

        print("d/dx    {}".format(d_dx))
        print("d/dy    {}".format(d_dy))
        print("d/dz    {}".format(d_dz))
        print("d2/dx2  {}".format(d2_dx2))
        print("d2/dy2  {}".format(d2_dy2))
        print("d2/dz2  {}".format(d2_dz2))
        # import ast
        # t = ast.parse(s)
        #
        # print(ast.dump(t, include_attributes=True))

        import math

        pi4 = math.pi / 2.0
        print("expr {}\nvalue {}".format(
            expr,
            expr.evalf(subs={shift: 0.0, x: pi4, y: pi4, z: pi4})
        ))


if __name__ == '__main__':
    import math
    pi4 = math.pi / 2.0
    ProblemP4.symbolic_version(Vector(pi4, pi4, pi4))