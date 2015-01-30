from __future__ import division

__author__ = 'nzhang-dev'

import numpy as np
import itertools

from hpgmg.finite_volume.space import Coord, Space
from hpgmg.finite_volume.mesh import Mesh


def tuple_multiply(tup, n):
    return tuple(map(lambda x: x * n, tup))


def tuple_add(tup, n):
    return tuple(map(lambda x: x + n, tup))


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
    new_mesh = Mesh(mesh.space().double_space())
    # new_mesh.fill(0)
    indices = Coord(mesh.shape).foreach()

    for index in indices:
        new_coord = index*2
        value = mesh[index]
        for delta in iter_delta(new_coord, new_mesh):
            target = new_coord + delta
            norm = np.linalg.norm(delta, 1)
            new_mesh[target] += (value * np.exp2(-norm))

    return new_mesh


def multi_iter(matrix):
    iterator = np.nditer(matrix, flags=['multi_index'])
    while not iterator.finished:
        yield iterator.multi_index
        iterator.iternext()


def legal_neighbors(point, shape):
    dimension_values = map(lambda x: (x-1, x, x+1), shape)
    return [
        pt
        for pt in itertools.product(*dimension_values)
        if pt.in_space(pt, shape) and pt != point
    ]


def restrict(mesh):
    new_mesh = Mesh(Coord(mesh.shape).halve_space())

    for index in mesh.indices():
        target_index = index * 2
        neighbors = legal_neighbors(index, mesh.shape)
        neighbor_mean = sum(mesh[x] for x in neighbors)/len(neighbors)
        new_mesh[target_index] = 0.5*mesh[index] + 0.5*neighbor_mean

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
    b = np.random.random(space)
    restrict(b)
    x = np.zeros(2**i)

if __name__ == '__main__':
    import sys
    main(sys.argv)