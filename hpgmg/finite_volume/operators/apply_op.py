from __future__ import print_function
from hpgmg.finite_volume.operators.specializers.apply_op_specializer import OclApplyOpSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class ApplyOp(object):
    def __new__(cls, *args, **kwargs):
        return ApplyOp.apply(*args, **kwargs)

    @staticmethod
    def apply(level, target_mesh, source_mesh):
        level.solver.boundary_updater.apply(level, source_mesh)
        with level.timer("apply_op"):
            ApplyOp.apply_interior(level, target_mesh, source_mesh)
            # for index in level.interior_points():
            #     target_mesh[index] = level.solver.problem_operator.apply_op(source_mesh, index, level)

    @staticmethod
    @specialized_func_dispatcher({
        'ocl': OclApplyOpSpecializer
    })
    def apply_interior(level, target_mesh, source_mesh):
        for index in level.interior_points():
            target_mesh[index] = level.solver.problem_operator.apply_op(source_mesh, index, level)