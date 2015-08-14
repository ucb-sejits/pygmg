from __future__ import print_function
from abc import ABCMeta, abstractmethod
from hpgmg.finite_volume.operators.specializers.interpolate_specializer import CInterpolateSpecializer, \
    OclInterpolateSpecializer
from hpgmg.finite_volume.operators.specializers.util import time_this, specialized_func_dispatcher

from hpgmg.finite_volume.space import Space, Coord


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Interpolator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def interpolate(self, target_level, target_mesh, source_mesh, ):
        pass


class InterpolatorPC(Interpolator):
    """
    interpolates by sampling
    """
    def __init__(self, solver, pre_scale):
        self.solver = solver
        self.dimensions = solver.dimensions
        self.pre_scale = pre_scale

    @time_this
    @specialized_func_dispatcher({
        'c': CInterpolateSpecializer,
        'omp': CInterpolateSpecializer,
        'ocl': OclInterpolateSpecializer
    })
    def interpolate(self, target_level, target_mesh, source_mesh):
        for target_index in target_level.interior_points():
            source_index = ((target_index - target_level.ghost_zone) // 2) + target_level.ghost_zone
            target_mesh[target_index] *= self.pre_scale
            target_mesh[target_index] += source_mesh[source_index]


class InterpolatorPQ(Interpolator):
    def __init__(self, solver, prescale=1.0):
        self.solver = solver
        self.pre_scale = prescale

        assert self.solver.dimensions == 3, "PQ interpolator currently only works in 3d"
        #
        # the interpolation is based on by associating a coefficient with a neighbor of
        # a position in the grid being interpolated
        self.convolution = [
            [-27.0, Coord(-1, -1, -1)],
            [270.0, Coord(0, -1, -1)],
            [45.0, Coord(+1, -1, -1)],
            [270.0, Coord(-1, 0, -1)],
            [-2700.0, Coord(0, 0, -1)],
            [-450.0, Coord(+1, 0, -1)],
            [45.0, Coord(-1, +1, -1)],
            [-450.0, Coord(0, +1, -1)],
            [-75.0, Coord(+1, +1, -1)],
            [270.0, Coord(-1, -1, 0)],
            [-2700.0, Coord(0, -1, 0)],
            [-450.0, Coord(+1, -1, 0)],
            [-2700.0, Coord(-1, 0, 0)],
            [27000.0, Coord(0, 0, 0)],
            [4500.0, Coord(+1, 0, 0)],
            [-450.0, Coord(-1, +1, 0)],
            [4500.0, Coord(0, +1, 0)],
            [750.0, Coord(+1, +1, 0)],
            [45.0, Coord(-1, -1, +1)],
            [-450.0, Coord(0, -1, +1)],
            [-75.0, Coord(+1, -1, +1)],
            [-450.0, Coord(-1, 0, +1)],
            [4500.0, Coord(0, 0, +1)],
            [750.0, Coord(+1, 0, +1)],
            [-75.0, Coord(-1, +1, +1)],
            [750.0, Coord(0, +1, +1)],
            [125.0, Coord(+1, +1, +1)],
        ]
        # but we want the given neighbor coord to be either backward or forward looking depending on whether
        # the grid index for that dimension is odd or even respectively
        # to do this we we convert each coord above into an array of 8 coords where each index is flipped
        # depending on the evenness of the source interpolation point

        self.neighbor_directions = [InterpolatorPQ.compute_neighbor_direction(point) for point in Space(2, 2, 2).points]
        self.convolution = [
            (row[0], [row[1] * neighbor_direction for neighbor_direction in self.neighbor_directions])
            for row in self.convolution
        ]
        self.one_over_32_cubed = 1.0/(32**3)

    @staticmethod
    def compute_neighbor_direction(coord):
        return Coord([1 if c % 2 == 0 else -1 for c in coord])

    @staticmethod
    def compute_neighbor_index(vector):
            return (vector.i % 2) * 4 + (vector.j % 2) * 2 + (vector.k % 2)

    @time_this
    def interpolate(self, target_level, target_mesh, source_level, source_mesh, ):
        for target_index in target_mesh.space.points:
            source_index = target_index // 2
            target_mesh[target_index] *= self.pre_scale
            oddness_index = InterpolatorPQ.compute_neighbor_index(target_index)

            accumulator = 0
            for coefficient, neighbor_index_offsets in self.convolution:
                accumulator += coefficient * source_mesh[source_index + neighbor_index_offsets[oddness_index]]

            target_mesh[target_index] += self.one_over_32_cubed * accumulator
