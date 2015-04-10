from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.space import Space

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil


class Jacobi3D(Stencil):
    def __init__(self, alpha, beta):
        super(Jacobi3D, self).__init__(
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
    import logging
    logging.basicConfig(level=0)

    j = Jacobi3D(0.0, 1.0/6.0)

    mesh = Mesh(Space(4, 4, 4))

    for index in mesh.indices():
        mesh[index] = sum(list(index))

    mesh.print("old mesh")

    new_mesh = j(mesh).view(Mesh)

    new_mesh.print("New mesh")
