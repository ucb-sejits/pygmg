from __future__ import print_function
import itertools
from stencil_code.halo_enumerator import HaloEnumerator
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.simple_level import SimpleLevel

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
    def __init__(self, solver, level):
        assert(isinstance(solver, SimpleMultigridSolver))
        assert(isinstance(level, SimpleLevel))

        self.solver = solver
        self.level = level

        if self.solver.boundary_is_dirichlet:
            self.apply = self.apply_dirichlet
        elif self.solver.boundary_is_periodic:
            self.apply = self.apply_periodic

        self.border_types = {}
        self.num_border_types = self.solver.dimensions - 1

    def apply_dirichlet(self, mesh):
        assert(isinstance(mesh, Mesh))

        halo_iterator = HaloEnumerator(self.level.ghost_zone, mesh.space)

        def get_scale_and_neighbor(point):
            sign = 1.0
            neighbor = []
            for dim in range(mesh.ndim):
                x = point[dim]
                if x < self.level.ghost_zone[dim]:
                    sign = -sign
                    neighbor.append(x+1)
                elif x >= mesh.space[dim]-self.level.ghost_zone[dim]:
                    sign = -sign
                    neighbor.append(x-1)
                else:
                    neighbor.append(x)
            return sign, tuple(neighbor)

        for index in halo_iterator.fixed_surface_iterator():

            scale, neighbor_index = get_scale_and_neighbor(index)
            mesh[index] = scale * mesh[neighbor_index]

    def apply_periodic(self, mesh):
        assert(isinstance(mesh, Mesh))

        halo_iterator = HaloEnumerator(self.level.ghost_zone, mesh.space)

        def get_scale_and_neighbor(point):
            neighbor = []
            for dim in range(mesh.ndim):
                x = point[dim]
                if x < self.level.ghost_zone[dim]:
                    neighbor.append(mesh.space[dim]-x-2)
                elif x >= mesh.space[dim]-self.level.ghost_zone[dim]:
                    neighbor.append(self.level.ghost_zone[dim] - (x - mesh.space[dim] + 1))
                else:
                    neighbor.append(x)
            return tuple(neighbor)

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
        ordered_border_types = sorted(
            list(itertools.product('ie', repeat=self.solver.dimensions))[1:],
            key=num_edges
        )
        for key, group in itertools.groupby(ordered_border_types, num_edges):
            self.border_types[key] = group


if __name__ == '__main__':
    simple_solver = SimpleMultigridSolver.get_solver(["2"])
    bu = BoundaryUpdaterV1(simple_solver, simple_solver.fine_level)
    print(bu.ordered_border_type_enumerator())
