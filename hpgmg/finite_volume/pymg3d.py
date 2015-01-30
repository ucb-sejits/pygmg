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


def interpolate(mesh):
    assert isinstance(mesh, Mesh)

    new_mesh = Mesh.from_coord(mesh.space().double())

    # expand known values
    for index in mesh.indices():
        new_mesh[index*2] = mesh[index]

    # interpolate to the center of each 'x' of known values
    for index in mesh.indices():
        target = index * 2 + 1

    max_index = tuple_add(new_shape, -1)
    for index in multi_iter(new_mesh):
        if point_in_shape(index, max_index):

    for i, j in itertools.product(range(1, new_shape, 2), repeat=2):
        new_mesh[i, j] = sum(
            new_mesh[m,n] for m, n in itertools.product((i-1, i+1), (j-1, j+1))
        ) / 4.0  # number of neighbors

    for i in range(new_shape):
        for j in range(1 - (i & 1), 2*(s-1)+1, 2):
            neighbors = [
                new_mesh[m, n]
                for m, n in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
                if 0 <= m < new_shape and 0 <= n < new_shape
            ]
            new_val = sum(neighbors) / len(neighbors)
            new_mesh[i, j] = new_val

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
    new_space = Coord.from_tuple(mesh.shape).halve()
    new_mesh = np.zeros(new_space.to_tuple())

    for index in mesh.indices():
        target_index = tuple_multiply(index, 2)
        neighbors = legal_neighbors(index, mesh.shape)
        neighbor_mean = sum(mesh[x] for x in neighbors)/len(neighbors)
        new_mesh[target_index] = 0.5*mesh[index] + 0.5*neighbor_mean

    return new_mesh


def multigridv(T, b, x):
    """
    :param b: numpy matrix
    :param x: numpy matrix
    :return: numpy matrix
    """
    i = b.shape[0]
    if i == 2**1 + 1:
        #compute exact
        return np.linalg.solve(T, b)
    x = smooth(b, x)
    residual = np.dot(T, x) - b
    diff = interpolate(multigridv(T, restrict(residual)), np.zeros_like(b))
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