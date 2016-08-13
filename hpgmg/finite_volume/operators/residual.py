from __future__ import print_function

import numpy as np
import snowflake.nodes as snodes
from snowflake.vector import Vector

from hpgmg.finite_volume.operators.specializers.smooth_specializer import CSmoothSpecializer, OmpSmoothSpecializer, \
    OclSmoothSpecializer, OclResidualSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher, time_this

import hpgmg.finite_volume

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

    def residue(self, level, target_mesh, source_mesh, right_hand_side, lambda_mesh):
        kern = self.get_residue_kernel(level)
        kern(target_mesh, source_mesh, right_hand_side)

    __residue_cache = {}
    def get_residue_kernel(self, level):
        if level in self.__residue_cache:
            return self.__residue_cache[level]
        component = level.kernel.get_stencil()
        sten = snodes.Stencil(
            snodes.StencilComponent("rhs", snodes.SparseWeightArray(
                {Vector.zero_vector(len(level.space)) : 1}
            )) - component,
            "target",
            ((1, -1, 1),)*len(level.space)
        )
        kern = self.__residue_cache[level] = hpgmg.finite_volume.compiler.compile(sten)
        return kern
