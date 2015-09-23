from __future__ import print_function

from stencil_code.neighborhood import Neighborhood

from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.operators.specializers.rebuild_specializer import CRebuildSpecializer, OclRebuildSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher, profile
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class StencilVonNeumannR1(BaseOperator):
    """
    implements a stencil using a radius 1 von neumann neighborhood
    i.e. 7 point in 3d
    """
    def __init__(self, solver):
        self.solver = solver
        self.a = solver.a
        self.b = solver.b
        self.dimensions = solver.dimensions
        self.restrictor = solver.restrictor
        self.is_variable_coefficient = solver.is_variable_coefficient
        self.h2inv = 1.0
        self.ghost_zone = Coord(1 for _ in range(self.dimensions))
        self.neighborhood_offsets = [
            Coord(x)
            for x in Neighborhood.von_neuman_neighborhood(radius=1, dim=self.dimensions, include_origin=False)
        ]
        self.unit_vectors = [
            Coord([1 if d == dim else 0 for d in range(self.dimensions)])
            for dim in range(solver.dimensions)
        ]
        self.num_neighbors = len(self.neighborhood_offsets)

        if solver.is_variable_coefficient:
            if solver.is_helmholtz:
                self.apply_op = self.apply_op_variable_coefficient_boundary_conditions_helmholtz
            else:
                self.apply_op = self.apply_op_variable_coefficient_boundary_conditions_poisson
        else:
            self.apply_op = self.apply_op_constant_coefficient_boundary_conditions

    def set_scale(self, level_h):
        self.h2inv = 1.0 / (level_h ** 2)

    def apply_op_variable_coefficient_boundary_conditions_helmholtz(self, mesh, index, level):
        return self.a * level.alpha[index] * mesh[index] - self.b * self.h2inv * (
            sum(
                level.beta_face_values[dim][index] * (mesh[index - self.unit_vectors[dim]] - mesh[index])
                for dim in range(self.dimensions)
            ) +
            sum(
                level.beta_face_values[dim][index + self.unit_vectors[dim]] * (
                    (
                        mesh[index + self.unit_vectors[dim]] - mesh[index]
                    )
                )
                for dim in range(self.dimensions)
            )
        )

    def apply_op_variable_coefficient_boundary_conditions_poisson(self, mesh, index, level):
        return -self.b * self.h2inv * (
            sum(
                level.beta_face_values[dim][index] * (
                    (
                        mesh[index - self.unit_vectors[dim]] - mesh[index]
                    )
                )
                for dim in range(self.dimensions)
            ) +
            sum(
                level.beta_face_values[dim][index + self.unit_vectors[dim]] * (
                    (
                        mesh[index + self.unit_vectors[dim]] - mesh[index]
                    )
                )
                for dim in range(self.dimensions)
            )
        )

    def apply_op_constant_coefficient_boundary_conditions(self, mesh, index, _=None):
        return self.a * mesh[index] - self.b * self.h2inv * (
            sum([mesh[index + neighbor_offset] for neighbor_offset in self.neighborhood_offsets]) -
            mesh[index] * self.num_neighbors
        )

    def apply_op_constant_coefficient_boundary_conditions_verbose(self, mesh, index, _=None):
        neighbor_sum = sum([mesh[index + neighbor_offset] for neighbor_offset in self.neighborhood_offsets])
        m_i = mesh[index]
        second_term = neighbor_sum - m_i * self.num_neighbors
        print("apply_op h2inv {} neighbor_sum {} second_term {}".format(self.h2inv, neighbor_sum, second_term))
        return self.a * m_i - self.b * self.h2inv * second_term

    @profile
    def rebuild_operator(self, target_level, source_level=None):
        self.set_scale(target_level.h)

        if source_level is not None:
            self.restrictor.restrict(target_level, target_level.alpha, source_level.alpha, Restriction.RESTRICT_CELL)
            for dim in range(self.dimensions):
                #print(np.sum(abs(target_level.beta_face_values[dim].ravel())))
                self.restrictor.restrict(target_level, target_level.beta_face_values[dim],
                                         source_level.beta_face_values[dim], dim+1)
                #print(np.sum(abs(target_level.beta_face_values[dim].ravel())))

        with target_level.timer("blas1"):
            target_level.dominant_eigen_value_of_d_inv_a = self.get_dominant_eigenvalue(target_level)

    @specialized_func_dispatcher({
        'c': CRebuildSpecializer,
        'omp': CRebuildSpecializer,
        'ocl': OclRebuildSpecializer
    })
    def get_dominant_eigenvalue(self, target_level):
        adjust_value = 2.0
        dominant_eigenvalue = -1e9
        for index in target_level.interior_points():
            if self.is_variable_coefficient:
                # print("VARIABLE COEFFICIENT")
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
                a_diagonal = self.a * target_level.alpha[index] - self.b * self.h2inv * (
                    sum(
                        target_level.beta_face_values[dim][index] *
                        (target_level.valid[index-self.unit_vectors[dim]] - adjust_value)
                        for dim in range(self.dimensions)
                    ) +
                    sum(
                        target_level.beta_face_values[dim][index + self.unit_vectors[dim]] *
                        (target_level.valid[index + self.unit_vectors[dim]] - adjust_value)
                        for dim in range(self.dimensions)
                    )
                )
            else:
                # print("CONSTANT COEFFICIENT")
                sum_abs = abs(self.b * self.h2inv) * sum(
                    target_level.valid[index + neighbor_offset] for neighbor_offset in self.neighborhood_offsets
                )
                a_diagonal = self.a - self.b * self.h2inv * sum(
                    target_level.valid[index + neighbor_offset]-adjust_value
                    for neighbor_offset in self.neighborhood_offsets
                )
                # print("##,{}:{:10.4f},{:10.4f},{:10.4f}".format(
                #     ",".join(map("{:02d}".format, index)),
                #     self.a * target_level.alpha[index], self.b*self.h2inv, a_diagonal
                # ))

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
        return dominant_eigenvalue