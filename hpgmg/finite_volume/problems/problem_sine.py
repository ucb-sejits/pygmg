from __future__ import print_function
from math import sin, cos

import numpy as np

from hpgmg.other_work.constants import Constants
import hpgmg.finite_volume.operators.misc as misc
from hpgmg.finite_volume.problems.problem import Problem
from hpgmg.finite_volume.space import Vector

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class SineProblem(Problem):
    alpha = 1.0
    beta = 1.0

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

                alpha = SineProblem.alpha
                beta = SineProblem.beta

                if is_variable_coefficient:
                    beta_i, _ = SineProblem.evaluate_beta(absolute_position-Vector(level.h*0.5, 0.0, 0.0))
                    beta_j, _ = SineProblem.evaluate_beta(absolute_position-Vector(0.0, level.h*0.5, 0.0))
                    beta_k, beta = SineProblem.evaluate_beta(absolute_position-Vector(0.0, 0.0, level.h*0.5))
                    beta, beta_xyz = SineProblem.evaluate_beta(absolute_position)
                    beta_ijk = Vector(beta_i, beta_j, beta_k)
                else:
                    beta_xyz = Vector(0.0, 0.0, 0.0)
                    beta_ijk = Vector(1.0, 1.0, 1.0)

                u, u_xyz, u_xxyyzz = SineProblem.evaluate_u(absolute_position)
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
