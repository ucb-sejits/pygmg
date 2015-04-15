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
                    self.weight * lambda_mesh[index] * (rhs_mesh[index] - self.operator(rhs_mesh, level.valid, index))
                )

            temp_mesh = working_target
            level.temp = level.target_mesh
            level.target_mesh = temp_mesh


class SejitsJacobiSmoother(Stencil):
    def __init__(self, alpha, beta):
        super(SejitsJacobiSmoother, self).__init__(
            backend='c',
            boundary_handling='clamp',
            neighborhoods=[Neighborhood.von_neuman_neighborhood(radius=1, dim=3, include_origin=False)]
        )
        self.alpha = alpha
        self.beta = beta

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = self.alpha * in_grid[x]
            for y in in_grid.neighbors(x, 0):
                out_grid[x] += self.beta * in_grid[y]


if __name__ == '__main__':
    if True:
        j = JacobiSmoother()
        mesh = Mesh(Space(4, 4, 4))
        for index in mesh.indices():
            mesh[index] = sum(list(index))
        mesh.print("mesh")

        smoothed_mesh = Mesh(mesh.space)

        JacobiSmoother.smooth(mesh, smoothed_mesh, 0.0, 1.0/6.0)

        smoothed_mesh.print("smoothed mesh")
    else:
        import logging
        logging.basicConfig(level=0)

        j = SejitsJacobiSmoother(0.0, 1.0/6.0)

        mesh = Mesh(Space(4, 4, 4))

        for index in mesh.indices():
            mesh[index] = sum(list(index))

        mesh.print("old mesh")

        new_mesh = Mesh(mesh.space)

        j(mesh, new_mesh)

        new_mesh.print("New mesh")
