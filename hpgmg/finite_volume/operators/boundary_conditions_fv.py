from __future__ import print_function
import functools
import itertools
import numpy as np

from stencil_code.halo_enumerator import HaloEnumerator

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.boundary_kernels.dirichlet import DirichletBoundary
from hpgmg.finite_volume.operators.boundary_kernels.periodic import PeriodicBoundary
from hpgmg.finite_volume.operators.specializers.boundary_specializer import CBoundarySpecializer, OmpBoundarySpecializer
from hpgmg.finite_volume.operators.specializers.util import time_this, specialized_func_dispatcher
from hpgmg.finite_volume.space import Vector


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
            self.name = "dirichlet"
            self.boundary = DirichletBoundary()
        elif self.solver.boundary_is_periodic:
            self.name = "periodic"
            self.boundary = PeriodicBoundary()

        self.kernels = [self.boundary.make_kernel(boundary) for boundary in self.boundary_cases()]

    @time_this
    @specialized_func_dispatcher({
        'c': CBoundarySpecializer,
        'omp': OmpBoundarySpecializer,
        'ocl': CBoundarySpecializer
    })
    def apply(self, level, mesh):
        for kernel in self.kernels:
            kernel(level, mesh)


    # #@time_this
    # @staticmethod
    # def apply_dirichlet(level, mesh):
    #     assert(isinstance(mesh, Mesh))
    #
    #     halo_iterator = HaloEnumerator(level.ghost_zone, mesh.space)
    #
    #     def get_scale_and_neighbor(point):
    #         sign = 1.0
    #         neighbor = []
    #         for dim in range(mesh.ndim):
    #             x = point[dim]
    #             if x < level.ghost_zone[dim]:
    #                 sign = -sign
    #                 neighbor.append(x+1)
    #             elif x >= mesh.space[dim]-level.ghost_zone[dim]:
    #                 sign = -sign
    #                 neighbor.append(x-1)
    #             else:
    #                 neighbor.append(x)
    #         return sign, tuple(neighbor)
    #
    #     with level.timer('apply_boundary'):
    #         for index in halo_iterator.fixed_surface_iterator():
    #             scale, neighbor_index = get_scale_and_neighbor(index)
    #             mesh[index] = scale * mesh[neighbor_index]
    #
    # #@time_this
    # @staticmethod
    # def apply_periodic(level, mesh):
    #     assert(isinstance(mesh, Mesh))
    #
    #     halo_iterator = HaloEnumerator(level.ghost_zone, mesh.space)
    #
    #     def get_scale_and_neighbor(point):
    #         neighbor = []
    #         for dim in range(mesh.ndim):
    #             x = point[dim]
    #             if x < level.ghost_zone[dim]:
    #                 neighbor.append(mesh.space[dim]-x-2)
    #             elif x >= mesh.space[dim]-level.ghost_zone[dim]:
    #                 neighbor.append(level.ghost_zone[dim] - (x - mesh.space[dim] + 1))
    #             else:
    #                 neighbor.append(x)
    #         return tuple(neighbor)
    #
    #     with level.timer('apply_boundary'):
    #         for index in halo_iterator.fixed_surface_iterator():
    #             neighbor_index = get_scale_and_neighbor(index)
    #             mesh[index] = mesh[neighbor_index]

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

    def boundary_cases(self):
        cases = [Vector(i) for i in itertools.product((0, -1, 1), repeat=self.solver.dimensions) if any(i)]
        cases.sort(key=functools.partial(np.linalg.norm))
        return cases