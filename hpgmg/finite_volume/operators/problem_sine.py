from __future__ import print_function
from math import tanh, sin, cos

import numpy as np
from hpgmg.finite_volume.constants import Constants

import hpgmg.finite_volume.operators.misc as misc
from hpgmg.finite_volume.operators.problem import Problem
from hpgmg.finite_volume.space import Vector

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ProblemInitializer(Problem):
    @staticmethod
    def evaluate_beta(vector):
        """
        :param vector: compute values for beta at point in vector
        :return:
        """
        
        x, y, z = vector

        b_min = 1.0
        b_max = 10.0
        c2 = (b_max-b_min)/2  # coefficients to affect this transition
        c1 = (b_max+b_min)/2
        c3 = 10.0  # how sharply (B)eta transitions
        x_center = 0.50
        y_center = 0.50
        z_center = 0.50
        # calculate distance from center of the domain (0.5, 0.5, 0.5)
        r2 = (x-x_center)**2 + (y-y_center)**2 + (z-z_center)**2
        r2x = 2.0*(x-x_center)
        r2y = 2.0*(y-y_center)
        r2z = 2.0*(z-z_center)

        r = r2**0.5
        rx = 0.5 * r2x * (r2 ** -0.5)
        ry = 0.5 * r2y * (r2 ** -0.5)
        rz = 0.5 * r2z * (r2 ** -0.5)
        # -------------------------
        beta = c1+c2*tanh(c3*(r-0.25))
        beta_x = c2*c3*rx * (1 - (tanh(c3*(r-0.25))**2))
        beta_y = c2*c3*ry * (1 - (tanh(c3*(r-0.25))**2))
        beta_z = c2*c3*rz * (1 - (tanh(c3*(r-0.25))**2))

        return beta, Vector(beta_x, beta_y, beta_z)

    @staticmethod
    def evaluate_u(coord):
        """
        compute the exact value of the function u for a given coordinate
        :param coord:
        :return: value of u and a tuple of u for each dimension
        """
        x, y, z = coord

        c1 = 2.0*np.pi
        c2 = 6.0*np.pi
        p = 13  # must be odd(?) and allows up to p-2 order MG
        u = (sin(c1*x)**p) * (sin(c1*y)**p) * (sin(c1*z)**p)
        u_x = c1*p*cos(c1*x) * (sin(c1*x)**(p-1)) * (sin(c1*y)**p) * (sin(c1*z)**p)
        u_y = c1*p*cos(c1*y) * (sin(c1*y)**(p-1)) * (sin(c1*x)**p) * (sin(c1*z)**p)
        u_z = c1*p*cos(c1*z) * (sin(c1*z)**(p-1)) * (sin(c1*x)**p) * (sin(c1*y)**p)

        u_xx = c1*c1*p * ((p-1) * (sin(c1*x)**(p-2)) * (cos(c1*x)**2) - sin(c1*x)**p) * (sin(c1*y)**p) * (sin(c1*z)**p)
        u_yy = c1*c1*p * ((p-1) * (sin(c1*y)**(p-2)) * (cos(c1*y)**2) - sin(c1*y)**p) * (sin(c1*x)**p) * (sin(c1*z)**p)
        u_zz = c1*c1*p * ((p-1) * (sin(c1*z)**(p-2)) * (cos(c1*z)**2) - sin(c1*z)**p) * (sin(c1*x)**p) * (sin(c1*y)**p)

        u += (sin(c2*x)**p) * (sin(c2*y)**p) * (sin(c2*z)**p)
        u_x += c2*p*cos(c2*x) * (sin(c2*x)**(p-1)) * (sin(c2*y)**p) * (sin(c2*z)**p)
        u_y += c2*p*cos(c2*y) * (sin(c2*y)**(p-1)) * (sin(c2*x)**p) * (sin(c2*z)**p)
        u_z += c2*p*cos(c2*z) * (sin(c2*z)**(p-1)) * (sin(c2*x)**p) * (sin(c2*y)**p)
        u_xx += c2*c2*p * ((p-1)*(sin(c2*x)**(p-2)) * (cos(c2*x)**2) - sin(c2*x)**p) * (sin(c2*y)**p) * (sin(c2*z)**p)
        u_yy += c2*c2*p * ((p-1)*(sin(c2*y)**(p-2)) * (cos(c2*y)**2) - sin(c2*y)**p) * (sin(c2*x)**p) * (sin(c2*z)**p)
        u_zz += c2*c2*p * ((p-1)*(sin(c2*z)**(p-2)) * (cos(c2*z)**2) - sin(c2*z)**p) * (sin(c2*x)**p) * (sin(c2*y)**p)

        return u, Vector(u_x, u_y, u_z), Vector(u_xx, u_yy, u_zz)

    @staticmethod
    def setup(level, h_level, a, b, is_variable_coefficient):
        level.h = h_level
        half_cell = Vector([0.5 for _ in level.element_space])

        for box in level.my_boxes:
            for element_index in level.box_space.points:
                absolute_position = (element_index + box.coord + half_cell) * level.h

                alpha = 1.0

                if is_variable_coefficient:
                    beta_i, _ = ProblemInitializer.evaluate_beta(absolute_position-Vector(level.h*0.5, 0.0, 0.0))
                    beta_j, _ = ProblemInitializer.evaluate_beta(absolute_position-Vector(0.0, level.h*0.5, 0.0))
                    beta_k, beta = ProblemInitializer.evaluate_beta(absolute_position-Vector(0.0, 0.0, level.h*0.5))
                    beta, beta_xyz = ProblemInitializer.evaluate_beta(absolute_position)
                    beta_ijk = Vector(beta_i, beta_j, beta_k)
                else:
                    beta = 1.0
                    beta_xyz = Vector(0.0, 0.0, 0.0)
                    beta_ijk = Vector(1.0, 1.0, 1.0)

                u, u_xyz, u_xxyyzz = ProblemInitializer.evaluate_u(absolute_position)
                f = a * alpha * u - (
                    b * (
                        (beta_xyz.i * u_xyz.i + beta_xyz.j * u_xyz.j + beta_xyz.k * u_xyz.k) +
                        beta * (u_xxyyzz.i + u_xxyyzz.j + u_xxyyzz.k)
                    )
                )

                box.vectors[Constants.VECTOR_UTRUE][element_index] = u
                box.vectors[Constants.VECTOR_ALPHA][element_index] = alpha
                box.vectors[Constants.VECTOR_F][element_index] = f
                box.vectors[Constants.VECTOR_BETA_I][element_index] = beta_ijk.i
                box.vectors[Constants.VECTOR_BETA_J][element_index] = beta_ijk.j
                box.vectors[Constants.VECTOR_BETA_K][element_index] = beta_ijk.k

        if level.alpha_is_zero is None:
            level.alpha_is_zero = misc.dot(level, Constants.VECTOR_ALPHA, Constants.VECTOR_ALPHA) == 0

    @staticmethod
    def symbolic_version(coord):
        from sympy import diff, Symbol, sin
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        c1, c2, exp = Symbol('c1'), Symbol('c2'), Symbol('exp')

        expr = sin(c1*x)**exp * sin(c1*y)**exp * sin(c1*z)**exp + \
               sin(c2*x)**exp * sin(c2*y)**exp * sin(c2*z)**exp

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
            expr.evalf(subs={c1: math.pi, c2: math.pi*6.0, exp: 13, x: pi4, y: pi4, z: pi4})
        ))


if __name__ == '__main__':
    ProblemInitializer.symbolic_version(Vector(1,1,1))