from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.space import Space, Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil


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


class JacobiSmoother(object):
    neighbors = Neighborhood.von_neuman_neighborhood(radius=1, dim=3, include_origin=False)

    @staticmethod
    def smooth(in_grid, out_grid, a, b):
        # for index in out_grid.space.interior_points(Coord(1, 1, 1)):
        for index in out_grid.indices():
            out_grid[index] = a * in_grid[index]

            for neighbor_offset in JacobiSmoother.neighbors:
                neighbor_index = out_grid.space.clamp(index + neighbor_offset)
                out_grid[index] += b * in_grid[neighbor_index]

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
