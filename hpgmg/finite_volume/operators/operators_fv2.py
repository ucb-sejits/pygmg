from __future__ import print_function
from stencil_code.neighborhood import Neighborhood
from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class OperatorsFV2(BaseOperator):
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
            self.apply_op = self.apply_op_variable_coefficient
        else:
            self.apply_op = self.apply_op_constant_coefficient

    def set_scale(self, level_h):
        self.h2inv = 1.0 / (level_h ** 2)

    def apply_op_variable_coefficient(self, mesh, index, level):
        """
        returns the stencil computation at index,
        stencil uses beta values for neighbors below index for a given dimension
        then uses the beta values of the index+1 for neighbors above index in the given dimension
        :param mesh:
        :param index:
        :param level:
        :return:
        """
        face_contributions = 0
        for dim in mesh.dim_range():
            # add contributions through lower face
            face_contributions += level.beta_face_values[dim][index] * (
                mesh[index - self.unit_vectors[dim]] - mesh[index]
            )
            # add contributions through upper face
            face_contributions += level.beta_face_values[dim][index + self.unit_vectors[dim]] * (
                mesh[index + self.unit_vectors[dim]] - mesh[index]
            )

        return self.a * level.alpha[index] * mesh[index] - self.b * self.h2inv * face_contributions

    def apply_op_constant_coefficient(self, mesh, index, _=None):
        """
        compute sum of neighbors - number of neighbors * value at index
        :param mesh:
        :param index:
        :param _: unused
        :return:
        """
        neighbor_sum = 0
        for neighbor_offset in self.neighborhood_offsets:
            neighbor_sum += mesh[index - neighbor_offset]
        return self.a * mesh[index] - self.b * self.h2inv * (
            neighbor_sum - mesh[index] * self.num_neighbors
        )

    def rebuild_operator(self, target_level, source_level=None):
        self.set_scale(target_level.h)

        if source_level is not None:
            self.restrictor.restrict(target_level, target_level.alpha, source_level.alpha, Restriction.RESTRICT_CELL)
            for dim in range(self.dimensions):
                self.restrictor.restrict(target_level, target_level.beta_face_values[dim],
                                         source_level.beta_face_values[dim], dim+1)

        adjust_value = 2.0
        dominant_eigenvalue = -1e9

        with target_level.timer("blas1"):
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
                        [target_level.valid[index + neighbor_offset] for neighbor_offset in self.neighborhood_offsets]
                    )
                    a_diagonal = self.a - self.b * self.h2inv * sum(
                        [
                            target_level.valid[index + neighbor_offset]-adjust_value
                            for neighbor_offset in self.neighborhood_offsets
                        ]
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
                target_level.dominant_eigen_value_of_d_inv_a = dominant_eigenvalue

