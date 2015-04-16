from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.smoothers import jacobi_stencil
from hpgmg.finite_volume.space import Space, Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil


class JacobiSmoother(object):
    def __init__(self, op, use_l1_jacobi=True, iterations=10):
        """

        :param op:
        :param use_l1_jacobi:
        :param iterations:
        :return:
        """
        self.operator = op
        self.use_l1_jacobi = use_l1_jacobi
        self.weight = 1.0 if use_l1_jacobi else 2.0/3.0
        self.iterations = iterations

    def smooth(self, level, target_mesh, rhs_mesh):
        """

        :param level: the level being smoothed
        :param target_mesh:
        :param rhs_mesh:
        :return:
        """
        lambda_mesh = level.l1_inverse if self.use_l1_jacobi else level.d_inverse
        working_target = level.temp
        for i in range(self.iterations):
            # jacobi_stencil(self.operator, lhs_mesh, rhs_mesh)
            for index in working_target.indices():
                working_target[index] = rhs_mesh[index] + (
                    self.weight * lambda_mesh[index] * (
                        rhs_mesh[index] - self.operator.apply_op(rhs_mesh, index)
                    )
                )

            temp_mesh = working_target
            level.temp = target_mesh
            working_target = temp_mesh


if __name__ == '__main__':
    import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

    solver = simple_hpgmg.SimpleMultigridSolver.get_solver("3 -fb".split())

    assert(
        isinstance(solver.smoother, JacobiSmoother),
        "solver.smoother {} is not a JacobiSmoother".format(solver.smoother)
    )

    base_level = solver.fine_level
    mesh = base_level.cell_values
    for index in mesh.indices():
        mesh[index] = sum(list(index))
    mesh.print("mesh")

    solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
    base_level.cell_values.print("smoothed mesh")
