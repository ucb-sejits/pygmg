from __future__ import print_function
from stencil_code.library.basic_convolution import Neighborhood
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class SimpleConstantCoefficientOperator(object):
    def __init__(self, solver):
        self.a = solver.a
        self.b = solver.b
        self.restrictor = solver.restrictor
        self.h2inv = 1.0
        self.neighborhood = [
            Coord(x)
            for x in Neighborhood.von_neuman_neighborhood(radius=1, dim=solver.dimensions, include_origin=False)
        ]
        self.num_neighbors = len(self.neighborhood)

    def set_scale(self, level_h):
        self.h2inv = 1.0 / (level_h ** 2)

    def apply_op(self, mesh, index):
        return self.a * mesh[index] - self.b * self.h2inv * sum(
            [mesh[neighbor_index] for neighbor_index in self.neighborhood]
        ) * self.num_neighbors

    def rebuild_operator(self, target_level, source_level=None):
        self.set_scale(target_level.h)
        self.restrictor.restrict(target_level, target_level.alpha, source_level.alpha, Restriction.RESTRICT_CELL)
        self.restrictor.restrict(target_level, target_level.beta_face_values[SimpleLevel.FACE_I],
                                 source_level.beta_face_values[SimpleLevel.FACE_I], Restriction.RESTRICT_FACE_I)
        self.restrictor.restrict(target_level, target_level.beta_face_values[SimpleLevel.FACE_J],
                                 source_level.beta_face_values[SimpleLevel.FACE_J], Restriction.RESTRICT_FACE_J)
        self.restrictor.restrict(target_level, target_level.beta_face_values[SimpleLevel.FACE_K],
                                 source_level.beta_face_values[SimpleLevel.FACE_K], Restriction.RESTRICT_FACE_K)

        dominant_eigenvalue = -1e9
        for index in target_level.interior_points():
            sum_abs = abs(self.b * self.h2inv) * sum(
                [target_level.valid[neighbor_index] for neighbor_index in self.neighborhood]
            )
            a_diagonal = self.a * target_level.alpha[index] - self.b * self.h2inv * sum(
                [target_level.valid[neighbor_index]-2.0 for neighbor_index in self.neighborhood]
            )

            # compute the d_inverse, l1_inverse and dominant eigen_value
            target_level.d_inverse[index] = 1.0/a_diagonal
            # as suggested by eq 6.5 in Baker et al,
            # "Multigrid smoothers for ultra-parallel computing: additional theory and discussion"..
            # formerly inverse of the L1 row norm... L1inv = ( D+D^{L1} )^{-1}
            if a_diagonal >= 1.5 * sum_abs:
                target_level.l1_inverse[index] = 1.0 / a_diagonal
            else:
                target_level.l1_inverse[index] = 1.0 / (a_diagonal + 0.5 * sum_abs)
            # upper limit to Gershgorin disc == bound on dominant eigenvalue
            di = (a_diagonal + sum_abs) / a_diagonal
            if di > dominant_eigenvalue:
                dominant_eigenvalue = di
