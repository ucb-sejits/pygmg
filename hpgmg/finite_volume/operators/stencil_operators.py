from __future__ import print_function

import numpy as np
from abc import ABCMeta, abstractmethod

from stencil_code.neighborhood import Neighborhood
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.smoothers import jacobi, jacobi_stencil
from hpgmg.finite_volume.space import Space, Coord

#from stencil_code.neighborhood import Neighborhood


__author__ = 'Shiv Sundram shivsundram@berkeley.edu U.C. Berkeley'

class Operator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply_op(self, x, i, j, k):
        pass

    def __mul__(self, x):
        if isinstance(x, Mesh):
            dim = x.space[0]
            x_new = np.copy(x)
            # FIX ME. currently x.space.dim return dimension-1, not dimension
            for i in range(1, dim-1):  # FIX ME thus upper bds should eventually be dim-1, not dim
                for j in range(1, dim-1):
                    for k in range(1, dim-1):
                        val = self.apply_op(x, i, j, k)
                        x_new[i][j][k] = val

            return x_new
        return NotImplemented


# this class adds -4*x[i][j] to the sum so as to make the laplacian diagonally dominant
class Weird6ptSum2d(Operator):
    def apply_op(self, x, i, j):
        return (x[i + 1][j] +
            x[i - 1][j] +
            x[i][j + 1] +
            x[i][j - 1] -
            x[i][j]*4
            )

    def sumAbsAij(self, valid, i, j):
        return (
            + valid[i][j + 1]
            + valid[i][j - 1]
            + valid[i + 1][j]
            + valid[i - 1][j])

    def Aii(self, x, i, j):
        length = x.space[0]
        if (not ((0<i and i<length-1) and (0<j and j<length-1) )):
            return 1
        else:
            return -4


    def Dinv(self, x, i, j):
        length = x.space[0]
        if (not ((0<i and i<length-1) and (0<j and j<length-1) )):
            return 1
        else:
            return float(1) / (-4)

    def constructMatrix(self, x):
        length = x.space[0]
        #print (length)
        matrix = Mesh((length**2, length**2))
        matrix.fill(0)
        for i in range(0, length):
            for j in range(0, length):
                ij = i*length +j
                if not  ((0<i and i<length-1) and (0<j and j<length-1) ):
                    matrix[ij][ij] = 1
                else:
                    neighbors = neighbors2d(i, j)
                    #print(neighbors)
                    for neighbor in neighbors:
                        ij1 = length*neighbor.i + neighbor.j
                        matrix[ij][ij1] = 1
                    matrix[ij][ij] = -4
        return matrix

    def __mul__(self, x):
        if isinstance(x, Mesh):
            dim = x.space[0]
            x_new = np.copy(x)
            # FIX ME. currently x.space.dim return dimension-1, not dimension
            for i in range(1, dim-1):  # FIX ME thus upper bds should eventually be dim-1, not dim
                for j in range(1, dim-1):
                        val = self.apply_op(x, i, j)
                        x_new[i][j] = val

            return x_new
        return NotImplemented


class Sum6pt2d(Operator):
    def __init__(self):
        pass

    def apply_op(self, x, i, j):
        return (x[i + 1][j] +
            x[i - 1][j] +
            x[i][j + 1] +
            x[i][j - 1]
            )

    def sumAbsAij(self, valid, i, j):
        return (
            + valid[i][j + 1]
            + valid[i][j - 1]
            + valid[i + 1][j]
            + valid[i - 1][j])

    def Aii(self, valid, i, j):
        return 0

    def constructMatrix(self, x):
        length = x.space[0]
        #print (length)
        matrix = Mesh((length**2, length**2))
        matrix.fill(0)
        for i in range(0, length):
            for j in range(0, length):
                ij = i*length +j
                if not  ((0<i and i<length-1) and (0<j and j<length-1) ):
                    matrix[ij][ij] = 1
                else:
                    neighbors = neighbors2d(i, j)
                    #print(neighbors)
                    for neighbor in neighbors:
                        ij1 = length*neighbor.i + neighbor.j
                        matrix[ij][ij1] = 1
        return matrix


    def __mul__(self, x):
        if isinstance(x, Mesh):
            dim = x.space[0]
            x_new = np.copy(x)
            # FIX ME. currently x.space.dim return dimension-1, not dimension
            for i in range(1, dim-1):  # FIX ME thus upper bds should eventually be dim-1, not dim
                for j in range(1, dim-1):
                        val = self.apply_op(x, i, j)
                        x_new[i][j] = val

            return x_new
        return NotImplemented

def neighbors2d(i, j):
    orgin = Coord(i, j)
    neighbs = [Coord] * 4
    deltas = [Coord(0, 1), Coord(1, 0), Coord(-1, 0), Coord(0, -1)]
    for i in range(0, len(deltas)):
        neighbs[i] = orgin+deltas[i]
    return neighbs


class ConstantCoefficient7pt(Operator):
    def __init__(self, a, b, dimensions=3, h2inv=1.0):
        self.a = a
        self.b = b
        self.h2inv = h2inv
        self.neighborhood = [
            Coord(x)
            for x in Neighborhood.von_neuman_neighborhood(radius=1, dim=dimensions, include_origin=False)
        ]

    def set_scale(self, level_h):
        self.h2inv = 1.0 / (level_h ** 2)

    def apply_op1d(self, x, i, j, k):
        j_stride = int(round(pow(len(x), 1.0 / 3)))  # dimension of grid with ghost regions
        k_stride = j_stride ** 2
        ijk = (1 + i) + (1 + j) * j_stride + (1 + k) * k_stride  # map i,j,k to ghost grid ijk
        return self.a * x[ijk] - self.b * self.h2inv * (
            + x[ijk + 1]
            + x[ijk - 1]
            + x[ijk + j_stride]
            + x[ijk - j_stride]
            + x[ijk + k_stride]
            + x[ijk - k_stride]
            - x[ijk] * 6.0
        )

    def apply_op(self, x, i, j, k):
        return self.a * x[i][j][k] - self.b * self.h2inv * (
            + x[i + 1][j][k]
            + x[i - 1][j][k]
            + x[i][j + 1][k]
            + x[i][j - 1][k]
            + x[i][j][k + 1]
            + x[i][j][k - 1]
            - x[i][j][k] * 6.0
        )


    def D_inv1d(self, x, i, j, k):
        # FIX ME. should simply use mesh methods to retrieve number of neighbors
        j_stride = int(round(pow(len(x), 1.0 / 3)))  #dimension of grid with ghost regions
        k_stride = j_stride ** 2
        ijk = (1 + i) + (1 + j) * j_stride + (1 + k) * k_stride  #map i,j,k to ghost grid ijk
        return 1.0 / (self.a - self.b * self.h2inv * (
            + valid[ijk - 1]
            + valid[ijk - j_stride]
            + valid[ijk - k_stride]
            + valid[ijk + 1]
            + valid[ijk + j_stride]
            + valid[ijk + k_stride]
            - 12.0
        ))

    def D_inv(self, x, i, j, k):
        # FIX ME. should simply use mesh methods to retrieve number of neighbors
        i, j, k = i + 1, j + 1, k + 1
        return 1.0 / (self.a - self.b * self.h2inv * (
            + valid[i][j][k - 1]
            + valid[i][j][k + 1]
            + valid[i][j + 1][k]
            + valid[i][j - 1][k]
            + valid[i + 1][j][k]
            + valid[i - 1][j][k]
            - 12.0
        ))


def initialize_valid_region1d(x):
    dim = int(round(pow(len(x), 1.0 / 3)))
    j_stride = dim
    k_stride = dim ** 2
    valid = np.zeros(len(x))

    for ind in range(0, len(x)):
        valid[ind] = 0

    for i in range(1, dim - 1):
        for j in range(1, dim - 1):
            for k in range(1, dim - 1):
                ijk = i + j * j_stride + k * k_stride
                valid[ijk] = 1
    return valid


def get_side_length(x):
    #assert x.shape[0] == x.shape[1] and x.shape[1] == x.shape[2], "Grid must be cubic"
    return x.shape[0]


def initialize_valid_region(x):
    dim = get_side_length(x)
    valid_shape = (dim + 1, dim + 1, dim + 1)  # dimensions of new valid cube
    valid = np.zeros(valid_shape)

    for i in range(0, dim):  # initialize all cells to 0/invalid
        for j in range(0, dim):
            for k in range(0, dim):
                valid[i][j][k] = 0

    for i in range(1, dim - 1):  # initialize non-ghost zone cells to 1, assuming ghost zone has width of 1
        for j in range(1, dim - 1):
            for k in range(1, dim - 1):
                valid[i][j][k] = 1
    return valid


def add_constant_boundary1d(x):
    old_dim = int(round(pow(len(x), 1.0 / 3)))
    j_stride = old_dim
    k_stride = old_dim ** 2

    new_dim = old_dim + 2
    new_j_stride = new_dim
    new_k_stride = new_dim ** 2

    new_x = np.zeros(new_dim ** 3)
    for i in range(0, old_dim):
        for j in range(0, old_dim):
            for k in range(0, old_dim):
                old_ijk = i + j * j_stride + k * k_stride
                new_ijk = (1 + i) + (1 + j) * new_j_stride + (1 + k) * new_k_stride
                new_x[new_ijk] = x[old_ijk]
    return new_x


def add_periodic_boundary(x):
    old_dim = get_side_length(x)
    new_dim = 1 + old_dim + 1  # there are ghost zones on both sides of cube
    new_x = Mesh((new_dim, new_dim, new_dim))  # dimensions of new cube
    return NotImplemented


def add_constant_boundary(x, value = 0):
    old_dim = get_side_length(x)
    new_dim = 1 + old_dim + 1  # there are ghost zones on both sides of cube
    new_x = Mesh((new_dim, new_dim, new_dim))  # dimensions of new cube
    new_x.fill(value)
    for i in range(0, old_dim):
        for j in range(0, old_dim):
            for k in range(0, old_dim):
                new_i, new_j, new_k = 1 + i, 1 + j, 1 + k  # map coordinate of old cube to new cube
                new_x[new_i][new_j][new_k] = x[i][j][k]
    return new_x

def add_constant_2d_boundary(x, value = 0):
    old_dim = get_side_length(x)
    new_dim = 1 + old_dim + 1  # there are ghost zones on both sides of cube
    new_x = Mesh((new_dim, new_dim))  # dimensions of new cube
    new_x.fill(value)
    for i in range(0, old_dim):
        for j in range(0, old_dim):
                new_i, new_j= 1 + i, 1 + j  # map coordinate of old cube to new cube
                new_x[new_i][new_j] = x[i][j]
    return new_x


def lin_space(x):
    c = 0
    dim = x.shape[0]
    for i in range(0, dim):
        for j in range(0, dim):
            for k in range(0, dim):
                x[i][j][k] = c
                c += 1

def lin_space2d(x):
    c = 0
    dim = x.shape[0]
    for i in range(0, dim):
        for j in range(0, dim):
                x[i][j] = c
                c += 1

if __name__ == "__main__":
    # xf = np.linspace(0.0, 63.0, num=64)
    #A = ConstantCoefficient7pt(1,2, .1)
    #print(A.apply_op1d(xf, 1,1,1))

    #print x
    #xf = np.linspace(0.0, 63.0, 64)
    #v = initialize_valid_region1d(xf)
    #print(v)

    #xf = np.linspace(0.0, 7.0, 8)
    #print(add_constant_boundary1d(xf))

    S = Weird6ptSum2d()
    xm = Mesh((2, 2))
    lin_space2d(xm)
    print("initial mesh without boundary")
    print(xm)
    xmb = add_constant_2d_boundary(xm, 0)
    #.1 for every boundary
    print("with boundary (xmb) of 1")
    print(xmb)
    Sm = S.constructMatrix(xmb)
    print("matrix of operator")
    print(Sm)
    print("result of applying stencil operator to mesh with boundary")
    stencil_result = S*xmb
    print (stencil_result)


    print ("result of applying matrix operator to mesh with boundary")
    matrix_mult_result = np.dot(Sm, xmb.flatten()).reshape(4, 4)
    print(matrix_mult_result)

    b = np.zeros_like(xmb.flatten())
    jacobi_np_result = jacobi(Sm, b, xmb.flatten(), 3)

    b = np.zeros_like(xmb)
    print("numpy jacobi result")
    print(jacobi_np_result.reshape(4, 4))

    jacobi_stencil_result = jacobi_stencil(S, b, xmb, 3)
    print("stencil jacobi result")
    print(jacobi_stencil_result)

    print("the two results should be the same")






