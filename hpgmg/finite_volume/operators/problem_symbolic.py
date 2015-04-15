from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from math import tanh, sin, cos

import numpy as np
from hpgmg.finite_volume.constants import Constants

import hpgmg.finite_volume.operators.misc as misc
from hpgmg.finite_volume.operators.problem import Problem
from hpgmg.finite_volume.operators.problem_sine import SineProblem
from hpgmg.finite_volume.space import Vector, Space

import sympy


class SymbolicProblem(Problem):
    """
    built for computing values for a function and it's first and second derivatives with
    respect to x, y, and z

    This class provides self.x (and y and z) as symbols, implementer can add more symbols as necessary
    as desired
    """

    x, y, z = sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')
    c1, c2 = 2.0 * np.pi, 6.0 * np.pi
    power = 13.0
    # c1, c2 = sympy.Symbol('c1'), sympy.Symbol('c2')
    # power = sympy.Symbol('power')

    u = (sympy.sin(c1*x)**power) * (sympy.sin(c1*y)**power) * (sympy.sin(c1*z)**power)

    print("u {}".format(u))
    print("du_dx {}".format(sympy.diff(u, x)))

    u = (sympy.sin(c1*x)**power) * (sympy.sin(c1*y)**power) * (sympy.sin(c1*z)**power) +\
        (sympy.sin(c2*x)**power) * (sympy.sin(c2*y)**power) * (sympy.sin(c2*z)**power)

    print("u {}".format(u))

    def __init__(self):
        self.u = SymbolicProblem.u

        self.du_dx = sympy.powsimp(sympy.diff(self.u, SymbolicProblem.x), force=True)
        self.du_dy = sympy.diff(self.u, SymbolicProblem.y)
        self.du_dz = sympy.diff(self.u, SymbolicProblem.z)

        self.d2u_dx2 = sympy.diff(self.u, SymbolicProblem.x, 2)
        self.d2u_dy2 = sympy.diff(self.u, SymbolicProblem.y, 2)
        self.d2u_dz2 = sympy.diff(self.u, SymbolicProblem.z, 2)

        print("du_dx {}".format(self.du_dx))

    @staticmethod
    def evaluate_beta(vector):
        pass

    def evaluate_u(self, coord):
        """
        compute the exact value of the function u for a given coordinate
        :param coord:
        :return: value of u and a tuple of u for each dimension
        """
        x, y, z = coord

        substitutions = {
            SymbolicProblem.x: coord.i,
            SymbolicProblem.y: coord.j,
            SymbolicProblem.z: coord.k,
            SymbolicProblem.c1: 2.0 * np.pi,
            SymbolicProblem.c2: 6.0 * np.pi,
            SymbolicProblem.power: 13.0,
        }


        u = self.u.evalf(subs=substitutions)

        u_x = float(self.du_dx.evalf(subs=substitutions))
        u_y = float(self.du_dy.evalf(subs=substitutions))
        u_z = float(self.du_dz.evalf(subs=substitutions))

        u_xx = float(self.d2u_dx2.evalf(subs=substitutions))
        u_yy = float(self.d2u_dy2.evalf(subs=substitutions))
        u_zz = float(self.d2u_dz2.evalf(subs=substitutions))

        return u, Vector(u_x, u_y, u_z), Vector(u_xx, u_yy, u_zz)


if __name__ == '__main__':
    sp = SymbolicProblem()

    mesh = Mesh(Space(10, 10, 10))
    if True:
        count = 0
        for index in mesh.indices():
            vector = Vector(float(index[d]) / mesh.space[d] for d in range(mesh.space.ndim))

            a, da, d2a = sp.evaluate_u(vector)
            b, db, d2b = SineProblem.evaluate_u(vector)

            do_print = False
            if abs(a-b) > 1e-6:
                print("mismatch u {} {}".format(a, b), end="")
                do_print = True
            if not da.near(db):
                print("mismatch du {} {}".format(da, db), end="")
                do_print = True
            if not d2a.near(d2b):
                print("mismatch d2u {} {}".format(d2a, d2b), end="")
                do_print = True
            if do_print:
                print()

            count += 1
            if count % 10 == 0:
                print("count {}".format(count))
    # SymbolicProblem.symbolic_version(Vector(1,1,1))



