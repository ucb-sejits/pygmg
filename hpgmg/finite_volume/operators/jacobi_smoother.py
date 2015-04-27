from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


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

    def smooth(self, level, mesh_to_smooth, rhs_mesh):
        """

        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        lambda_mesh = level.l1_inverse if self.use_l1_jacobi else level.d_inverse
        working_target = level.temp
        working_source = mesh_to_smooth
        need_copy = False

        for i in range(self.iterations):
            for index in level.interior_points():
                stencil = self.operator.apply_op(rhs_mesh, index, level)
                working_target[index] = working_source[index] + (
                    self.weight * lambda_mesh[index] * (
                        rhs_mesh[index] - stencil
                    )
                )

            temp = working_target
            working_target = working_source
            working_source = temp
            need_copy = not need_copy

        if need_copy:
            level.shift_mesh(mesh_to_smooth, 1.0, level.temp)


if __name__ == '__main__':
    import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

    solver = simple_hpgmg.SimpleMultigridSolver.get_solver("3 -fb".split())

    assert isinstance(solver.smoother, JacobiSmoother), "solver.smoother {} is not a JacobiSmoother".format(
        solver.smoother)


    base_level = solver.fine_level
    mesh = base_level.cell_values
    for point in mesh.indices():
        mesh[point] = sum(list(point))
    mesh.print("mesh")

    solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
    base_level.cell_values.print("smoothed mesh")
