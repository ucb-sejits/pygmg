from __future__ import print_function

import numpy as np

from hpgmg.finite_volume.operators.specializers.smooth_specializer import CSmoothSpecializer, OmpSmoothSpecializer, \
    OclSmoothSpecializer, OclResidualSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher, time_this


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Residual(object):
    def __init__(self, solver):
        self.solver = solver
        self.a = self.solver.a
        self.b = self.solver.b
        self.operator = self.solver.problem_operator

    def run(self, level, target_mesh, source_mesh, right_hand_side):
        self.solver.boundary_updater.apply(level, source_mesh)
        with level.timer("residual"):
            self.residue(level, target_mesh, source_mesh, right_hand_side, np.zeros((1,)))

    # noinspection PyUnusedLocal
    @specialized_func_dispatcher({
        'c': CSmoothSpecializer,
        'omp': OmpSmoothSpecializer,
        'ocl': OclResidualSpecializer
    })
    def residue(self, level, target_mesh, source_mesh, right_hand_side, lambda_mesh):
        for index in level.interior_points():
            a_x = self.operator.apply_op(source_mesh, index, level)
            target_mesh[index] = right_hand_side[index] - a_x
