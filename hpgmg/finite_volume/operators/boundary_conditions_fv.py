from __future__ import print_function
from stencil_code.halo_enumerator import HaloEnumerator
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class BoundaryUpdater_V1(object):
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

    def apply(self, mesh, just_faces):
        assert(isinstance(mesh, Mesh))

        halo_iterator = HaloEnumerator(self.level.ghost_zone, mesh.space)

        def get_scale_and_neighbor(index):
            scale = 1.0
            neighbor = []
            for dim in range(mesh.ndim):
                x = index[dim]
                if x < self.level.ghost_zone[dim]:
                    scale = -scale
                    neighbor.append(x+1)
                elif x >= mesh.space[dim]-self.level.ghost_zone[dim]:
                    scale = -scale
                    neighbor.append(x-1)
                else:
                    neighbor.append(x)
            return scale, tuple(neighbor)

        for index in halo_iterator.fixed_surface_iterator():

            scale, neighbor_index = get_scale_and_neighbor(index)
            mesh[index] = scale * mesh[neighbor_index]


if __name__ == '__main__':
    solver = SimpleMultigridSolver.get_solver(["2"])
    bu = BoundaryUpdater_V1(solver, solver.fine_level)

    for i in solver.fine_level.interior_points():
        solver.fine_level.cell_values[i] = i[0]

    solver.fine_level.cell_values.print("before bu")

    bu.apply(solver.fine_level.cell_values, False)

    solver.fine_level.cell_values.print("after bu")