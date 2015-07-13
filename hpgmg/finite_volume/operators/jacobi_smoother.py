from __future__ import print_function

from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.smoother import Smoother
from hpgmg.finite_volume.operators.specializers.inline_jit import partial_jit
from hpgmg.finite_volume.operators.specializers.smooth_specializer import CSmoothSpecializer, OmpSmoothSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher, profile, time_this

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

    #@time_this
    @partial_jit
    def smooth(self, level, mesh_to_smooth, rhs_mesh):
        """

        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        lambda_mesh = level.l1_inverse if self.use_l1_jacobi else level.d_inverse

        working_target, working_source = mesh_to_smooth, level.temp

        self.operator.set_scale(level.h)

        for i in range(self.iterations):
            working_target, working_source = working_source, working_target
            level.solver.boundary_updater.apply(level, working_source)
            self.smooth_points(level, working_source, working_target, rhs_mesh, lambda_mesh)

    @time_this
    @specialized_func_dispatcher({
        'c': CSmoothSpecializer,
        'omp': OmpSmoothSpecializer
    })
    def smooth_points(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        for index in level.interior_points():
            a_x = self.operator.apply_op(working_source, index, level)
            b = rhs_mesh[index]
            working_target[index] = working_source[index] + (self.weight * lambda_mesh[index] * (b - a_x))