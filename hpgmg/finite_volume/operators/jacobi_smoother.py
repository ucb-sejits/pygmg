from __future__ import print_function
from collections import namedtuple

from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.smoother import Smoother
from hpgmg.finite_volume.operators.specializers.smooth_specializer import jit_smooth

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class JacobiSmoother(Smoother):
    def __init__(self, op, use_l1_jacobi=True, iterations=6):
        """

        :param op:
        :param use_l1_jacobi:
        :param iterations:
        :return:
        """
        assert(isinstance(op, BaseOperator))
        assert(isinstance(use_l1_jacobi, bool))
        assert(isinstance(iterations, int))
        assert iterations % 2 == 0

        self.operator = op
        self.use_l1_jacobi = use_l1_jacobi
        self.weight = 1.0 if use_l1_jacobi else 2.0/3.0
        self.iterations = iterations

    def smooth(self, level, mesh_to_smooth, rhs_mesh):
        """

        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        rhs_mesh.dump("JACOBI_RHS_MESH")
        lambda_mesh = level.l1_inverse if self.use_l1_jacobi else level.d_inverse
        # if level.solver.dump_grids:
            # if lambda_mesh == level.l1_inverse:
            #     print("USING L1_INVERSE")
            # else:
            #     print("USING D_INV")
        working_target, working_source = mesh_to_smooth, level.temp
        lambda_mesh.dump("LAMBDA_MESH")

        self.operator.set_scale(level.h)
        #print(self.iterations)
        for i in range(self.iterations):
            working_target, working_source = working_source, working_target
            level.solver.boundary_updater.apply(level, working_source)

            working_source.dump("JACOBI_MESH_TO_SMOOTH_SOURCE")
            working_target.dump("JACOBI_MESH_TO_SMOOTH_TARGET")
            with level.timer("smooth"):
                self.smooth_points(level, working_source, working_target, rhs_mesh, lambda_mesh)
            #print(working_target)


                    # print("index {} Ax_n {} b {} lm {} w {} src {} trg {}".format(
                    #     ",".join(map(str,index)), a_x, b, lambda_mesh[index], self.weight,
                    #     working_source[index], working_target[index]
                    # ))

            working_target.dump("JACOBI_SMOOTH_PASS_{}_SIZE_{}".format(i, format(level.space[0]-2)))

    @jit_smooth
    def smooth_points(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        for index in level.interior_points():
            a_x = self.operator.apply_op(working_source, index, level)
            b = rhs_mesh[index]
            working_target[index] = working_source[index] + (self.weight * lambda_mesh[index] * (b - a_x))