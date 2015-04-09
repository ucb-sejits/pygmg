from __future__ import print_function
from hpgmg.finite_volume.space import Space

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from stencil_code.neighborhood import Neighborhood


class Interpolator(object):
    def __init__(self, prescale=1.0):
        self.pre_scale = prescale

    def interpolate(self, target_level, target_mesh, source_level, source_mesh, ):
        for target_index in target_mesh.space.points:
            pass


if __name__ == '__main__':
    n = Neighborhood.moore_neighborhood(radius=1, dim=3)

    def compute_neighbor_index(vector):
        return (vector.i % 2) * 4 + (vector.j % 2) * 2 + (vector.k % 2)

    for index in Space(2, 2, 2).points:
        print("{:3d}{:3d}{:3d}=>{:4d}  ".format(index.i, index.j, index.k, compute_neighbor_index(index)), end="")
    print()

    for i in n:
        print(i)
