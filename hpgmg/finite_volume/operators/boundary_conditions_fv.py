from __future__ import print_function
import itertools

from stencil_code.halo_enumerator import HaloEnumerator

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.specializers.util import time_this


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class BoundaryUpdaterV1(object):
    """
    For cell-centered, we need to fill in the ghost zones to apply any BC's
    This code does a simple linear interpolation for homogeneous dirichlet (0 on boundary)
    Nominally, this is first performed across faces, then to edges, then to corners.
    In this implementation, these three steps are fused

      . . . . . . . . . .        . . . . . . . . . .
      .       .       .          .       .       .
      .   ?   .   ?   .          .+x(0,0).-x(0,0).
      .       .       .          .       .       .
      . . . . +---0---+--        . . . . +-------+--
      .       |       |          .       |       |
      .   ?   0 x(0,0)|          .-x(0,0)| x(0,0)|
      .       |       |          .       |       |
      . . . . +-------+--        . . . . +-------+--
      .       |       |          .       |       |

    """
    def __init__(self, solver):
        self.solver = solver

        if self.solver.boundary_is_dirichlet:
            self.apply = BoundaryUpdaterV1.apply_dirichlet
            self.name = "dirichlet"
        elif self.solver.boundary_is_periodic:
            self.apply = BoundaryUpdaterV1.apply_periodic
            self.name = "periodic"

    #@time_this
    @staticmethod
    @time_this
    def apply_dirichlet(level, mesh):
        assert(isinstance(mesh, Mesh))

        halo_iterator = HaloEnumerator(level.ghost_zone, mesh.space)

        def get_scale_and_neighbor(point):
            sign = 1.0
            neighbor = []
            for dim in range(mesh.ndim):
                x = point[dim]
                if x < level.ghost_zone[dim]:
                    sign = -sign
                    neighbor.append(x+1)
                elif x >= mesh.space[dim]-level.ghost_zone[dim]:
                    sign = -sign
                    neighbor.append(x-1)
                else:
                    neighbor.append(x)
            return sign, tuple(neighbor)

        with level.timer('apply_boundary'):
            for index in halo_iterator.fixed_surface_iterator():
                scale, neighbor_index = get_scale_and_neighbor(index)
                mesh[index] = scale * mesh[neighbor_index]

    #@time_this
    @staticmethod
    @time_this
    def apply_periodic(level, mesh):
        assert(isinstance(mesh, Mesh))

        halo_iterator = HaloEnumerator(level.ghost_zone, mesh.space)

        def get_scale_and_neighbor(point):
            neighbor = []
            for dim in range(mesh.ndim):
                x = point[dim]
                if x < level.ghost_zone[dim]:
                    neighbor.append(mesh.space[dim]-x-2)
                elif x >= mesh.space[dim]-level.ghost_zone[dim]:
                    neighbor.append(level.ghost_zone[dim] - (x - mesh.space[dim] + 1))
                else:
                    neighbor.append(x)
            return tuple(neighbor)

        with level.timer('apply_boundary'):
            for index in halo_iterator.fixed_surface_iterator():
                neighbor_index = get_scale_and_neighbor(index)
                mesh[index] = mesh[neighbor_index]

    def ordered_border_type_enumerator(self):
        """
        border types are faces, edges, corners, hyper-corners, ...
        :return:
        """
        # TODO: Figure out some way to make use of this
        def num_edges(vec):
            return len(filter(lambda x: x == 'e', vec))
        return sorted(
            list(itertools.product('ie', repeat=self.solver.dimensions))[1:],
            key=num_edges
        )
