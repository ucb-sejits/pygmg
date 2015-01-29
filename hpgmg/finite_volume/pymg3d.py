from __future__ import division

__author__ = 'nzhang-dev'

import numpy as np
import itertools

from hpgmg.finite_volume.space import Coord, Space


def halve_space(space):
    return (space // 2) + 1


def double_space(space):
    return (space * 2) - 1



def tuple_multiply(tup, n):
    return tuple(map(lambda x: x * n, tup))


def tuple_add(tup, n):
    return tuple(map(lambda x: x + n, tup))


def smooth(b, x):
    return x


def interpolate(mesh):
    new_shape = double_shape(mesh.shape)

    new_mesh = np.zeros((new_shape,) * 2)

    # expand known values
    for index in multi_iter(mesh):
        expanded_index = tuple_multiply(index, 2)
        new_mesh[expanded_index] = mesh[index]

    # interpolate to the center of each 'x' of known values
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


def point_in_shape(point, shape):
    return all(
        map(
            lambda x: x[0] < x[1],
            zip(point, shape)
        )
    )

def legal_neighbors(point, shape):
    dimension_values = list(map(lambda x: (x-1, x, x+1), shape))
    return [
        point
        for point in itertools.product(*dimension_values)
        if point_in_shape(point, shape)
    ]


def restrict(mesh):
    new_space = halve_space(Space(mesh.shape))
    new_mesh = np.zeros(new_space)

    for index in multi_iter(mesh):
        target_index = tuple_multiply(index, 2)
        neighbors = legal_neighbors(index, mesh.shape)
        for neighbor_index in neighbors:
            new_mesh[target_index] += mesh[neighbor_index]
        new_mesh[target_index] /= len(neighbors)

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