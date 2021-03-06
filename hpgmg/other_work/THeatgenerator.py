#author: Shiv Sundram

#HPGGMG is used to solve elliptic PDE's which are discretized and thus 
#instatiated as a Tu=f linear system. This file generates different forms of the T matrix
#matrix coefficients based off of finite difference methods
#only works for linear/square/cubic grids, but can be generalized to rectangular/prism grids

from mesh import Mesh
from space import Coord

import itertools
import numpy as np


#generates lapalacian matrix for solving poisson problem with 1d n sized mesh
import numpy as np
import math
def gen1DHeatMatrixL(n):
    T = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                T[i, j] = 2
            elif j == i-1 or j == i+1:
                T[i, j] = -1
    return T



#indexing for x in 2d mesh is as follows
# 6 7 8
# 3 4 5
# 0 1 2 

#generates laplacian matrix for solving poisson problem with 2d nxn sized mesh
def gen2DHeatMatrixL(n):
    T = np.zeros(shape=(n*n,n*n))
    for i in range(n*n):
        for j in range(n*n):
            if i == j:
                T[i, j]=4
            elif j == i-1 or j == i+1 or j == i-n or j == i+n:
                T[i, j] = -1
    return T


def generate_2d_laplacian(n, neighborhood=4, torus=False):
    """
    See http://en.wikipedia.org/wiki/Laplacian_matrix#Example_of_the_Operator_on_a_Grid
    :param n:
    :param neighborhood: number of points in neighborhood, must be 4 or 8
    :param torus: wrap adjacency matrix
    :return:
    """
    adjacency = np.zeros([n*n, n*n])
    if neighborhood == 8:
        dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    else:
        dx = [-1, 0, 1, 0]
        dy = [0, -1, 0, 1]

    for x in range(n):
        for y in range(n):
            index = x * n + y
            if torus:
                for ne in range(len(dx)):
                    new_x = ( x + dx[ne] ) % n
                    new_y = ( y + dy[ne] ) % n
                    index2 = new_x * n + new_y
                    adjacency[index, index2] = 1
            else:
                for ne in range(len(dx)):
                    new_x = x + dx[ne]
                    new_y = y + dy[ne]
                    if 0 <= new_x < n and 0 <= new_y < n:
                        index2 = new_x * n + new_y
                        adjacency[index, index2] = 1

    degree = np.diag(sum(adjacency, 2))  # Compute the degree matrix
    laplacian_operator = degree - adjacency  # Compute the laplacian matrix in terms of the degree and adjacency matrices
    return laplacian_operator


def demmel_laplacian(n):
    t = np.zeros([n*n, n*n])
    for i in range(n*n):
        t[i, i] = 2
        if 0 < i:
            t[i, i-1] = -1.0
        if i < n-2:
            t[i, i+1] = -1.0

    L = np.kron(t, np.identity(n*n)) + np.kron(np.identity(n*n), t)
    return L

def gen3DHeatMatrixL(n):
    """
    generates laplacian matrix for solving poisson problem with 3d nxnxn mesh

    :param n:
    :return:
    """
    T = np.zeros(shape=(n*n*n,n*n*n))
    for i in range(n*n*n):
        for j in range(n*n*n):
            if i == j:
                T[i, j] = 8
            elif j == i-1 or j == i+1 or j == i-n or j == i+n or j == i+n*n or j == i-n*n:
                T[i, j] = -1
    return T

#
def gen3DHeatMatrixT(n, C, delta, h):
    '''
    This method generates the correct matrix T for solving the 3D heat equation, 
    using the Laplacian L 

    :param n: size of 1 dimension of the nxnxn mesh
    :param C: thermal diffusivity constant
    :param delta: timestep size
    :param h: position step size
    :return: the matrix T for solving Tu=f to numerically solve heat equation
    '''
    L = gen3DHeatMatrixL(n)
    z=C*delta/(h*h)
    return np.identity(n*n*n) - z*L

################################
#Generating N-eigen_values Laplacian matrix

def laplacian(n, ndim):
    """
    :param n: length in each dimension of space
    :param ndim: number of dimensions of space
    :return: 2n-eigen_values reshaped to 2-eigen_values mesh, as per the matrix generation note
    """
    m = Mesh((n, )*ndim*2)
    reference_mesh = Mesh((n, )*ndim)
    diagonal_entry = 2 * ndim
    for coord in reference_mesh.indices():
        for other in reference_mesh.space.neighbors(coord, 1):
            target = Coord(tuple(coord) + tuple(other)) #concatenates them into a single coord with coord first and other second
            m[target] = -1
        diagonal_target = Coord(tuple(coord)*2)
        m[diagonal_target] = diagonal_entry
    return m.reshape((n**ndim,)*2)


def heat_matrix(n, C, delta, h, ndim):
    """
    :param n: length in each dimension
    :param C: Heat diffusivity constant
    :param delta: Timestep
    :param h: Gap between samples
    :param ndim: Number of dimensions
    :return: Matrix for parameters above
    """
    z = C * delta / (h*h)
    return np.identity(n**ndim) - z * laplacian(n, ndim)


class LaplacianStencil(object):

    def __mul__(self, other):
        output = Mesh(other.space)
        for index in other.indices():
            for neighbor_coord in other.space.neighbors(index, 1):
                output[index] += -1*other[neighbor_coord]
            output[index] *= 2 * other.ndim * other[index]
        return output

    __rmul__ = __mul__


def is_cholesky_factorizable(matrix):
    return np.linalg.cholesky(matrix) is not None


if __name__ == '__main__':
    l = demmel_laplacian(17)
    print gen3DHeatMatrixT(3, .4, .7, 2)




