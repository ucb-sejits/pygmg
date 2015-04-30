from __future__ import division, print_function

__author__ = 'nzhang-dev'

import numpy as np
import functools

from space import Coord
from mesh import Mesh
import cProfile


def cache(func):
    func_cache = {}
    sentinel = object()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = tuple(args) + tuple(sorted(kwargs.items()))
        res = func_cache.get(key, sentinel)
        if res is sentinel:
            return func_cache.setdefault(key, func(*args, **kwargs))
        return res
    return wrapper

def to_mesh(func):
    @functools.wraps(func)
    def wrapper(arg):
        return func(arg.view(Mesh))
    return wrapper

@cache
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

@cache
def restriction_matrix(ndim):
    result = interpolation_matrix(ndim) / np.exp2(ndim)
    result.setflags(write=False)
    return result

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


    #eigen_vectors cycle
    def MGV(self, A, b, x):
        """
        :param A: A in Ax=b
        :param b: b in Ax=b
        :param x: guess for x
        :return: refined guess for X in Ax = b
        """
        if min(b.shape) <= 3: # minimum size, so we compute exact
            return np.linalg.solve(A, b.flatten()).reshape(x.shape)

        x = self.smooth(A, b, x, self.smooth_iterations)
        residual = (np.dot(A, x.flatten()) - b.flatten()).reshape(x.shape)
        restricted_residual = self.restrict(residual)
        diff = self.interpolate(self(self._evolve(A, b.ndim), restricted_residual, np.zeros_like(restricted_residual)))
        x -= diff
        result = self.smooth(A, b, x, self.smooth_iterations)
        return result    


    #Full Multi-Grid
    def FMG(self, A, b, x=None):
        """
        :param A: A in Ax=b
        :param b: b in Ax=b
        :param x: guess for x
        :return: refined guess for X in Ax = b
        """
        matrices = []
        Amatrices = []
        b_restrict = b[:]
        A_restrict = A[:]
        matrices.append(b_restrict)
        Amatrices.append(A_restrict)

        #continuously restrict b, and store all intermediate b's in list
        while(not (min(b_restrict.shape) <= 3)):
            b_restrict = self.restrict(b_restrict)
            A_restrict = self._evolve(A_restrict, b.ndim)
            matrices.append(b_restrict)
            Amatrices.append(A_restrict)

        #base case
        shape = b_restrict.shape
        x = ( np.linalg.solve(A_restrict, b_restrict.flatten()) )
        x = x.reshape(shape)

        for i in range(len(matrices)-2,-1,-1):
            x = self.MGV(Amatrices[i], matrices[i], interpolate(x))
        return x




    def __call__(self, A, b, x, cycle="eigen_vectors"):
        if cycle=="eigen_vectors":
            return self.MGV(A, b, x) #FIXME
        elif cycle=="F":
            return self.FMG(A, b, x) #FIXME

        #too lazy to comment out all of this below
        if False:
            """
            :param A: A in Ax=b
            :param b: b in Ax=b
            :param x: guess for x
            :return: refined guess for X in Ax = b
            """

            if min(b.shape) <= 3: # minimum size, so we compute exact
                return np.linalg.solve(A, b.flatten()).reshape(x.shape)

            x = self.smooth(A, b, x, self.smooth_iterations)
            residual = (np.dot(A, x.flatten()) - b.flatten()).reshape(x.shape)
            restricted_residual = self.restrict(residual)
            diff = self.interpolate(self(self._evolve(A, b.ndim), restricted_residual, np.zeros_like(restricted_residual)))
            x -= diff
            result = self.smooth(A, b, x, self.smooth_iterations)
            return result




    @staticmethod
    def _evolve(A, ndim):
        n = int(round((A.shape[0])**(1/ndim)))
        n = int((n + 1)/2)
        slices = tuple(slice(0, n**ndim) for _ in range(A.ndim))
        return A[slices]



    def profiled_call(self, A, b, x):
        cProfile.runctx('self(A, b, x)', {'self': self, 'A':A, 'b':b, 'x':x}, {})



if __name__ == '__main__':
    n = 9
    dims = 3
    from hpgmg.other_work.THeatgenerator import gen3DHeatMatrixT
    from hpgmg.other_work.smoothers import gauss_siedel
    A = gen3DHeatMatrixT(9, 0.2, 1, 1/n)
    b = np.random.random((n,)*dims)
    x = np.zeros_like(b)
    solver = MultigridSolver(interpolate_m, restrict_m, gauss_siedel, 4)
    print(solver(A, b, x))