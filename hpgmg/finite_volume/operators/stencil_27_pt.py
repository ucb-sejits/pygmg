from __future__ import print_function
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


def stencil_get_radius():
    return 1


class Stencil27(object):
    def __init__(self):
        self.radius = 1.0
        self.is_star = False

        self.face_coefficient = -128.0/30.0
        self.edge_coefficient = 14.0/30.0
        self.corner_coefficient = 3/30.0
        self.center_coefficient = 1.0/30.0

        self.face_neighbors = [
            Coord(0, 0, -1), Coord(0, 0, 1), Coord(0, -1, 0), Coord(0, 1, 0), Coord(-1, 0, 0), Coord(1, 0, 0),
        ]
        self.edge_neighbors = [
            Coord(0, -1, -1), Coord(0, -1, 1), Coord(0, 1, -1), Coord(0, 1, 1),
            Coord(-1, 0, -1), Coord(-1, 0, 1), Coord(1, 0, -1), Coord(1, 0, 1),
            Coord(-1, -1, 0), Coord(-1, 1, 0), Coord(1, -1, 0), Coord(1, 1, 0),
        ]
        self.corner_neighbors = [
            Coord(-1, -1, -1), Coord(-1, -1, 1), Coord(-1, 1, -1), Coord(-1, 1, 1),
            Coord(1, -1, -1), Coord(1, -1, 1), Coord(1, 1, -1), Coord(1, 1, 1),
        ]
        assert(len(self.face_neighbors)+len(self.edge_neighbors)+len(self.corner_neighbors) == 26)

    @staticmethod
    def stencil_get_radius():
        return 1

    def get_radius(self):
        return self.radius

    def is_star_shaped(self):
        return self.is_star

    def rebuild_operator(self, target_level, source_level, a, b):
        """
        Need to figure out this exact spec

        :param target_level:
        :param source_level:
        :param a:
        :param b:
        :return:
        """
        inverse_of_h_squared = 1.0 / (target_level.h**2)
        block_eigen_value = -1.0e9
        dominant_eigen_value = -1.0e9

        for index in target_level.indices():
            # radius of Gershgorin disc is the sum of the absolute values of the off-diagonal elements...
            sum_abs_a_ij = (
                abs(b * 6.0 * self.face_coefficient) +
                abs(b * 12.0 * self.edge_coefficient) +
                abs(b * 8.0 * self.corner_coefficient)
            ) * inverse_of_h_squared
            # center of Gershgorin disc is the diagonal element...
            a_ii = a - ( b * inverse_of_h_squared * self.center_coefficient)

            self.d_inverse[index] = 1.0 / a_ii
            if a_ii if a_ii >= 1.5 * sum_abs_a_ij:
                self.l1_inverse[index] = 1.0 / a_ii
            else:
                self.l1_inverse[index] = 1.0 / ( a_ii + 0.5 * sum_abs_a_ij)

            di = (a_ii + sum_abs_a_ij) / a_ii
            if di > block_eigen_value:
                block_eigen_value = di

        if block_eigen_value > dominant_eigen_value:
            dominant_eigen_value = block_eigen_value

        target_level.dominant_eigen_value_of_d_inv_a = dominant_eigen_value