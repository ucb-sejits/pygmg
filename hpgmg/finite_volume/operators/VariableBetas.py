from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from math import tanh
from hpgmg.finite_volume.space import Vector

class VariableBeta(object):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.center = Vector(0.5 for _ in range(self.dimensions))

    def evaluate_beta(self, vector):
        """
        :param vector: compute values for beta at point in vector
        :return:
        """

        b_min = 1.0
        b_max = 10.0
        c2 = (b_max-b_min)/2  # coefficients to affect this transition
        c1 = (b_max+b_min)/2
        c3 = 10.0  # how sharply (B)eta transitions

        delta = vector - self.center
        distance = sum(d**2 for d in delta)**0.5
        normalized_delta = delta * distance**-1.0

        beta_at_face = c1 + c2 * tanh(c3 * (distance - 0.25))
        beta_vector = c2 * c3 * normalized_delta * (1 - (tanh(c3 * (distance - 0.25))**2))

        return beta_at_face, beta_vector

    def evaluate_beta_3d(self, vector):
        """
        :param vector: compute values for beta at point in vector
        :return:
        """

        assert self.dimensions == 3, "evaluate_beta_3d only works for 3d problems"

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


if __name__ == '__main__':
    from hpgmg.finite_volume.mesh import Mesh

    size = 16
    num_dimensions = 3
    h = 1.0 / size

    variable_beta = VariableBeta(dimensions=num_dimensions)

    mesh = Mesh([size for _ in range(num_dimensions)])

    for coord in mesh.indices():
        index = coord.to_vector()

        a_beta_d, a_beta = variable_beta.evaluate_beta(index)
        b_beta_d, b_beta = variable_beta.evaluate_beta_3d(index)

        assert  abs(a_beta_d - b_beta_d) < 1e-6, \
            "at {} a_beta_d did not match b_beta_d {} != {}".format(index, a_beta_d, b_beta_d)
        for dim in range(num_dimensions):
            assert a_beta.near(b_beta), \
                "at {} a_beta_d did not match b_beta_d {} != {}".format(index, a_beta[dim], b_beta[dim])

        for d in range(num_dimensions):
            index_at_face_d = Vector(
                v if d != dim else v - 0.5 * h for dim, v in enumerate(index)
            )

            b_beta_d, b_beta = variable_beta.evaluate_beta_3d(index_at_face_d)
            a_beta_d, a_beta = variable_beta.evaluate_beta(index_at_face_d)

            assert abs(a_beta_d - b_beta_d) < 1e-6, \
                "at {} a_beta_d did not match b_beta_d {} != {}".format(index_at_face_d, a_beta_d, b_beta_d)
            assert a_beta.near(b_beta), \
                "at {} a_beta_d did not match b_beta_d {} != {}".format(index_at_face_d, a_beta, b_beta)
