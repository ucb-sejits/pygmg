from __future__ import division

__author__ = 'nzhang-dev'

import numpy as np
import itertools

from space import Coord, Space
from mesh import Mesh
from smoothers import gausssiedel as smooth

def interpolation_matrix(ndim):
    """
    :param ndim: Int, number of dimensions
    :return: np.ndarray so of weights
    """
    weight_matrix = np.zeros((3,)*ndim).view(Mesh)
    coord = Coord(1,1,1)  # center of matrix
    for delta in weight_matrix.space.neighbor_deltas():
        new_coord = coord + delta
        norm = np.linalg.norm(delta, 1)
        weight_matrix[new_coord] = np.exp2(-norm)
    weight_matrix.setflags(write=False)
    return weight_matrix

def interpolate(mesh):
    new_mesh = Mesh(mesh.space * 2)

    for index in mesh.indices():
        new_coord = index*2
        value = mesh[index]
        for delta in new_mesh.space.neighbor_deltas():
            target = new_coord + delta
            if target in new_mesh:
                norm = np.linalg.norm(delta, 1)
                new_mesh[target] += (value * np.exp2(-norm))
    return new_mesh

def interpolate_m(mesh):
    """
    :param mesh: Mesh to be interpolate
    :return: interpolated matrix

    Using matrix embedding and multiply instead of neighbor iteration
    """
    new_matrix = np.zeros((mesh.space + 1)*2).view(Mesh)
    inter_matrix = interpolation_matrix(mesh.ndim)
    for coord in mesh.indices():
        target = coord * 2 + 1
        neighborhood_slice = new_matrix.space.neighborhood_slice(target)
        new_matrix[neighborhood_slice] += mesh[coord] * inter_matrix
    return new_matrix[(slice(1,-1),)*mesh.ndim]


def restrict(mesh):
    new_mesh = Mesh(mesh.space/2)
    for index in new_mesh.indices():
        target = index*2
        weights = 0
        value = 0
        for delta in mesh.space.neighbor_deltas():
            neighbor = target + delta
            if neighbor in mesh.space:
                weight = np.exp2(-(np.linalg.norm(delta, 1) + mesh.space.ndim))
                value += mesh[neighbor] * weight
                weights += weight
        new_mesh[index] = value / weights
    return new_mesh

def simple_restrict(mesh):
    slices = tuple(slice(None, None, 2) for _ in range(mesh.ndim))
    return mesh[slices]


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
    import cProfile
    space = Space([int(args[1])]*int(args[2]))
    print("hello world")
    i = 4
    b = np.random.random(space).view(Mesh)
    for func in (restrict, simple_restrict, interpolate, interpolate_m):
        print(func.__name__)
        cProfile.runctx('{}(b)'.format(func.__name__), {'b': b, func.__name__: func}, {})


if __name__ == '__main__':
    import sys
    main(sys.argv)