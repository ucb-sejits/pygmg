#author: Shiv Sundram

#HPGGMG is used to solve elliptic PDE's which are discretized and thus 
#instatiated as a Tu=f linear system. This file generates different forms of the T matrix
#matrix coefficients based off of finite difference methods
#only works for linear/square/cubic grids, but can be generalized to rectangular/prism grids



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



# sample indexing for 3x3x3
#     / 24 25 26 /
#    / 21 22 23 /	
#   / 18 19 20 /

#     / 15 16 17 /
#    / 12 13 14 /	
#   /  9 10 11 /

#     /  6  7  8 /
#    /  3  4  5 /	
#   /  0  1  2 /

#generates laplacian matrix for solving poisson problem with 3d nxnxn mesh
def gen3DHeatMatrixL(n):
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

print gen3DHeatMatrixT(3, .4, .7, 2)




