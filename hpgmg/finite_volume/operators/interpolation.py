from __future__ import print_function
from hpgmg.finite_volume.space import Space, Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from stencil_code.neighborhood import Neighborhood


class Interpolator(object):
    def __init__(self, prescale=1.0):
        self.pre_scale = prescale
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
        # but we want the a given neighbor coord to be either backward or forward looking depending on whether
        # the grid index for that dimension is odd or even respectively
        # to do this we we convert each coord above into an array of 8 coords where each index is flipped
        # depending on the evenness of the source interpolation point

        self.neighbor_directions = [Interpolator.compute_neighbor_direction(point) for point in Space(2, 2, 2).points]
        self.convolution = [
            [(row[0], [row[1] * neighbor_direction for neighbor_direction in self.neighbor_directions])]
            for row in self.convolution
        ]

    @staticmethod
    def compute_neighbor_direction(coord):
        return Coord([1 if c % 2 == 0 else -1 for c in coord])

    @staticmethod
    def compute_neighbor_index(vector):
            return (vector.i % 2) * 4 + (vector.j % 2) * 2 + (vector.k % 2)

    def interpolate(self, target_level, target_mesh, source_level, source_mesh, ):
        for target_index in target_mesh.space.points:
            source_index = target_index // 2
            neighbor_index_offset = Interpolator.compute_neighbor_index(target_index)
            for coefficient, neighbor_index_offsets in self.convolution:
                target_mesh[target_index] *= self.pre_scale
                target_mesh[target_index] += coefficient * source_mesh[
                    source_index + neighbor_index_offsets[neighbor_index_offset]
                ]


if __name__ == '__main__':
    n = Neighborhood.moore_neighborhood(radius=1, dim=3)

    def compute_neighbor_index(vector):
        return (vector.i % 2) * 4 + (vector.j % 2) * 2 + (vector.k % 2)

    for index in Space(2, 2, 2).points:
        print("{:3d}{:3d}{:3d}=>{:4d}  ".format(index.i, index.j, index.k, compute_neighbor_index(index)), end="")
    print()

    # for i in n:
    #     print(i)

    interpolator = Interpolator(1.0)
    print("nd {}".format([(x, i) for x, i in enumerate(interpolator.neighbor_directions)]))
    print(interpolator.convolution[0])
    print(interpolator.convolution[1])