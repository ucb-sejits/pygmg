from __future__ import print_function
from stencil_code.neighborhood import Neighborhood
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class StencilVonNeumannR1(object):
    """
    implements a stencil using a radius 1 von neumann neighborhood
    """
    # TODO: implement variable coefficient cases
    def __init__(self, solver):
        self.solver = solver
        self.a = solver.a
        self.b = solver.b
        self.dimensions = solver.dimensions
        self.restrictor = solver.restrictor
        self.is_variable_coefficient = solver.is_variable_coefficient
        self.h2inv = 1.0
        self.neighborhood = [
            Coord(x)
            for x in Neighborhood.von_neuman_neighborhood(radius=1, dim=self.dimensions, include_origin=False)
        ]
        self.unit_vectors = [
            Coord([1 if d == dim else 0 for d in range(self.dimensions)])
            for dim in range(solver.dimensions)
        ]
        self.num_neighbors = len(self.neighborhood)
        self.apply_op = self.apply_op_constant_coefficient_unfused_boundary_conditions

    def set_scale(self, level_h):
        self.h2inv = 1.0 / (level_h ** 2)

    def apply_op_variable_coefficient_fused_boundary_conditions_helmholtz(self, mesh, index, level):
        return self.a * mesh[index] - self.b * self.h2inv * (
            sum(
                level.beta_face_values[SimpleLevel.FACE_I][index] * (
                    level.valid[index - self.unit_vectors[dim]] * (
                        mesh[index] + mesh[index - self.unit_vectors[dim]]
                    ) - 2.0 * mesh[index]
                )
                for dim in range(self.dimensions)
            ) +
            sum(
                level.beta_face_values[SimpleLevel.FACE_I][index + self.unit_vectors[dim]] * (
                    level.valid[index + self.unit_vectors[dim]] * (
                        mesh[index] + mesh[index + self.unit_vectors[dim]]
                    ) - 2.0 * mesh[index]
                )
                for dim in range(self.dimensions)
            )
        )

    def apply_op_variable_coefficient_fused_boundary_conditions_poisson(self, mesh, index, level):
        return -self.b * self.h2inv * (
            sum(
                level.beta_face_values[SimpleLevel.FACE_I][index] * (
                    level.valid[index - self.unit_vectors[dim]] * (
                        mesh[index] + mesh[index - self.unit_vectors[dim]]
                    ) - 2.0 * mesh[index]
                )
                for dim in range(self.dimensions)
            ) +
            sum(
                level.beta_face_values[SimpleLevel.FACE_I][index + self.unit_vectors[dim]] * (
                    level.valid[index + self.unit_vectors[dim]] * (
                        mesh[index] + mesh[index + self.unit_vectors[dim]]
                    ) - 2.0 * mesh[index]
                )
                for dim in range(self.dimensions)
            )
        )

    def apply_op_constant_coefficient_fused_boundary_conditions(self, mesh, index, level):
        return self.a * mesh[index] - self.b * self.h2inv * (
            sum(
                level.valid[index + self.unit_vectors[dim]] * (
                    mesh[index] + mesh[index - self.unit_vectors[dim]]
                )
                for dim in range(self.dimensions)
            ) +
            sum(
                level.valid[index + self.unit_vectors[dim]] * (
                    mesh[index] + mesh[index - self.unit_vectors[dim]]
                )
                for dim in range(self.dimensions)
            ) -
            mesh[index] * self.num_neighbors * 2.0
        )

    # TODO: Implement apply_op_variable_coefficient_unfused_boundary_conditions_helmholtz
    # TODO: Implement apply_op_variable_coefficient_unfused_boundary_conditions_poisson

    def apply_op_constant_coefficient_unfused_boundary_conditions(self, mesh, index, _):
        return self.a * mesh[index] - self.b * self.h2inv * (
            sum([mesh[neighbor_index] for neighbor_index in self.neighborhood]) -
            mesh[index] * self.num_neighbors
        )

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
            if self.is_variable_coefficient:
                # double sumAbsAij = fabs(b*h2inv) * (
                #                      fabs( beta_i[ijk        ]*valid[ijk-1      ] )+
                #                      fabs( beta_j[ijk        ]*valid[ijk-jStride] )+
                #                      fabs( beta_k[ijk        ]*valid[ijk-kStride] )+
                #                      fabs( beta_i[ijk+1      ]*valid[ijk+1      ] )+
                #                      fabs( beta_j[ijk+jStride]*valid[ijk+jStride] )+
                #                      fabs( beta_k[ijk+kStride]*valid[ijk+kStride] )
                #                    );
                #
                # // center of Gershgorin disc is the diagonal element...
                # double    Aii = a*alpha[ijk] - b*h2inv*(
                #                   beta_i[ijk        ]*( valid[ijk-1      ]-2.0 )+
                #                   beta_j[ijk        ]*( valid[ijk-jStride]-2.0 )+
                #                   beta_k[ijk        ]*( valid[ijk-kStride]-2.0 )+
                #                   beta_i[ijk+1      ]*( valid[ijk+1      ]-2.0 )+
                #                   beta_j[ijk+jStride]*( valid[ijk+jStride]-2.0 )+
                #                   beta_k[ijk+kStride]*( valid[ijk+kStride]-2.0 )
                #                 );
                sum_abs = abs(self.b * self.h2inv) * (
                    sum(
                        abs(
                            target_level.beta_face_values[dim][index] *
                            target_level.valid[index-self.unit_vectors[dim]]
                        )
                        for dim in range(self.dimensions)
                    ) +
                    sum(
                        abs(
                            target_level.beta_face_values[dim][index + self.unit_vectors[dim]] *
                            target_level.valid[index + self.unit_vectors[dim]]
                        )
                        for dim in range(self.dimensions)
                    )
                )
                # TODO: confirm the value of adjust for dimensions other than 3
                adjust_value = 2.0
                a_diagonal = self.a * target_level.alpha[index] - self.b * self.h2inv * (
                    sum(
                        target_level.beta_face_values[dim][index] *
                        target_level.valid[index-self.unit_vectors[dim]] - adjust_value
                        for dim in range(self.dimensions)
                    ) +
                    sum(
                        target_level.beta_face_values[dim][index + self.unit_vectors[dim]] *
                        target_level.valid[index + self.unit_vectors[dim]] - adjust_value
                        for dim in range(self.dimensions)
                    )
                )
            else:
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
