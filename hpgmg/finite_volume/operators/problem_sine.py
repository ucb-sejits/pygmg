from __future__ import print_function
from math import tanh, sin, cos
import numpy as np

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ProblemInitializer(object):
    @staticmethod
    def evaluate_beta(vector):
        """

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
        beta = c1+c2*tanh( c3*(r-0.25))
        beta_x = c2*c3*rx * (1 - (tanh(c3*(r-0.25))**2))
        beta_y = c2*c3*ry * (1 - (tanh(c3*(r-0.25))**2))
        beta_z = c2*c3*rz * (1 - (tanh(c3*(r-0.25))**2))

        return beta, (beta_x, beta_y, beta_z)

    @staticmethod
    def evaluate_u(coord, is_periodic):
        """
        compute the exact value of the function u for a given coordinate
        :param coord:
        :param is_periodic: 
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

        return u, (u_x, u_y, u_z), (u_xx, u_yy, u_zz)


    @staticmethod
    def setup(level, h_level, a, b):
        level.h = h_level


