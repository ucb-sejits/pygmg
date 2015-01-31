from __future__ import division

__author__ = 'nzhang-dev'

import numpy as np
import itertools

from space import Coord, Space
from mesh import Mesh

def smooth(b, x):
    return x


def iter_delta(coord, mesh):
    for direction in itertools.product((-1, 0, 1), repeat=3):
        delta = Coord(*direction)
        new_coord = delta + coord
        if all(0 <= dim < m for dim, m in zip(new_coord, mesh.shape)):
            yield delta


def interpolate(mesh):
    assert isinstance(mesh, Mesh)
    new_mesh = Mesh(mesh.space * 2)

    for index in mesh.indices():
        new_coord = index*2
        value = mesh[index]
        for delta in iter_delta(new_coord, new_mesh):
            target = new_coord + delta
            norm = np.linalg.norm(delta, 1)
            new_mesh[target] += (value * np.exp2(-norm))

    return new_mesh

def restrict(mesh):
    new_mesh = Mesh(mesh.space/2)
    for index in new_mesh.indices():
        target = index+1
        for delta in mesh.space.neighbor_deltas():
            neighbor = target + delta
            if neighbor in mesh.space:
                weight = np.exp2(-(np.linalg.norm(delta, 1) + mesh.space.ndim))
                new_mesh[index] += mesh[neighbor] * weight
    return new_mesh


def multi_grid_v_cycle(t, b, x):
    """
    :param b: numpy matrix
    :param x: numpy matrix
    :return: numpy matrix
    """
    i = b.shape[0]
    if i == 2**1 + 1:
        #compute exact
        return np.linalg.solve(t, b)
    x = smooth(b, x)
    residual = np.dot(t, x) - b
    diff = interpolate(multi_grid_v_cycle(t, restrict(residual)), np.zeros_like(b))
    x -= diff
    x = smooth(b,x)
    return x


def main(args):
    space = Space(64, 64, 64)

    print("hello world")
    i = 4
    b = np.random.random(space).view(Mesh)
    restricted = restrict(b)
    x = np.zeros(2**i)
    print(restricted)()

if __name__ == '__main__':
    import sys
    main(sys.argv)