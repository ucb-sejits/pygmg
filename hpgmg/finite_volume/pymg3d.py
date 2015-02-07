from __future__ import division, print_function

__author__ = 'nzhang-dev'

import numpy as np
import itertools

from space import Coord, Space
from mesh import Mesh
from smoothers import smooth_matrix as smooth
import cProfile

def to_mesh(func):
    def wrapper(arg):
        return func(arg.view(Mesh))
    return wrapper

def to_mesh(func):
    def wrapper(arg):
        return func(arg.view(Mesh))
    return wrapper

def to_mesh(func):
    def wrapper(arg):
        return func(arg.view(Mesh))
    return wrapper

def interpolation_matrix(ndim):
    """
    :param ndim: Int, number of dimensions
    :return: np.ndarray so of weights
    """
    weight_matrix = np.zeros((3,)*ndim).view(Mesh)
    coord = Coord((1,)*ndim)  # center of matrix
    for delta in weight_matrix.space.neighbor_deltas():
        new_coord = coord + delta
        norm = np.linalg.norm(delta, 1)
        weight_matrix[new_coord] = np.exp2(-norm)
    weight_matrix.setflags(write=False)
    return weight_matrix

def restriction_matrix(ndim):
    return interpolation_matrix(ndim) / np.exp2(ndim)

@to_mesh
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

@to_mesh
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

@to_mesh
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

@to_mesh
def restrict_m(mesh):
    """
    :param mesh: mesh to be restricted
    :return: new mesh, with boundary conditions copied
    """
    new_mesh = Mesh(mesh.space/2)
    ndim = len(new_mesh.space)
    restriction_mat = restriction_matrix(ndim)
    for index in new_mesh.indices():
        target = index * 2
        if new_mesh.space.is_boundary_point(index):
            new_mesh[index] = mesh[target]
        else:
            sub_matrix = mesh[mesh.space.neighborhood_slice(target)]
            new_value = np.tensordot(restriction_mat, sub_matrix, ndim)
            new_mesh[index] = new_value
    return new_mesh






@to_mesh
def simple_restrict(mesh):
    slices = tuple(slice(None, None, 2) for _ in range(mesh.ndim))
    return mesh[slices]


class MultigridSolver(object):
    def __init__(self, interpolate, restrict, smooth, smooth_iterations):
        self.interpolate = interpolate
        self.restrict = restrict
        self.smooth = smooth
        self.smooth_iterations = smooth_iterations

    def __call__(self, A, b, x):
        """
        :param A: A in Ax=b
        :param b: b in Ax=b
        :param x: guess for x
        :return: refined guess for X in Ax = b
        """
        if len(b.flatten()) <= 3: # minimum size, so we compute exact
            return np.linalg.solve(A, b)

        x = self.smooth(A, b, x, self.smooth_iterations)
        residual = (np.dot(A, x) - b)
        restricted_residual = self.restrict(residual)
        diff = self.interpolate(self(self.restrict(A), restricted_residual, np.zeros_like(restricted_residual)))
        x -= diff
        result = self.smooth(A, b, x, self.smooth_iterations)
        return result

    def profiled_call(self, A, b, x):
        cProfile.runctx('self(A, b, x)', {'self': self, 'A':A, 'b':b, 'x':x}, {})


def main(args):
    def _print_array(arr):
        if len(arr.shape) == 1:
            print(*arr, sep="\t"*1)
        else:
            for row in arr:
                _print_array(row)
    import smoothers, cProfile
    shape = Space((int(args[1]),)*int(args[2]))
    solution_shape = Space((int(args[1]),)*(int(args[2])-1))
    seed = sum(ord(i) for i in 'SEJITS')
    np.random.seed(seed)
    iterations = int(args[3])
    z = np.random.ranf()*32
    d1 = np.random.ranf()*z
    d2 = np.random.ranf()*(z-d1)
    # A = np.random.random(shape)
    # b = np.random.random(solution_shape)
    A = np.identity(shape[0])*(z)
    A += np.diag((d1,)*(shape[0]-1), 1)
    A += np.diag((d2,)*(shape[0]-1), -1)
    b = np.random.random((shape[0],))
    x = np.linalg.solve(A, b)
    solver = MultigridSolver(interpolate_m, restrict_m, smoothers.gauss_siedel, iterations)
    x_1 = solver(A, b, np.zeros_like(b))
    # solver.profiled_call(A, b, np.zeros_like(b))
    x_2 = smoothers.gauss_siedel(A, b, np.zeros_like(b), iterations*2)
    print()
    for name, result in (("numpy", x), ("MGS", x_1), ("Smoother", x_2)):
        print(name)
        print(*result)
        print()
        print("error")
        e = error(A, b, result)
        print()
        print("Absolute Relative Error")
        print(max(abs(err/val) for err, val in zip(e, b)))
        print()


if __name__ == '__main__':
    def error(A, b, x):
        return np.dot(A, x) - b
    import sys
    res = main(sys.argv)