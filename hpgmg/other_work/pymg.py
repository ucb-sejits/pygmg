from __future__ import division

__author__ = 'nzhang-dev'

import numpy as np
import itertools

def smooth(b, x):
    return x

def interpolate(mat):
    s = mat.shape[0]
    new_s = s*2 - 1
    new_array = np.zeros((new_s,) * 2)
    for i,j in itertools.product(range(s), repeat=2):
        new_array[2*i, 2*j] = mat[i,j]

    for i,j in itertools.product(range(1, new_s, 2), repeat=2):
        new_array[i,j] = sum(new_array[m,n] for m,n in itertools.product((i-1, i+1), (j-1, j+1))) / 4.0 #number of neighbors

    for i in range(new_s):
        for j in range(1 - (i & 1), 2*(s-1)+1, 2):
            neighbors = [new_array[m,n] for m,n in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
                         if (m >= 0 and m < new_s) and (n >= 0 and n < new_s)]
            new_val = sum(neighbors) / len(neighbors)
            new_array[i,j] = new_val

    return new_array


def restrict(mat):
    s = mat.shape[0]
    new_s = (s - 1) // 2 + 1
    new_matrix = np.zeros((new_s,) * 2)

    for i, j in itertools.product(range(new_s), repeat=2):
        m, n = 2 * i, 2 * j
        neighbors = [a for a in itertools.product((m - 1, m, m + 1), (n - 1, n, n + 1)) if a != (m, n)]
        clamped = [(a, b) for a, b in neighbors if (a >= 0) and (b >= 0) and (a < s) and (b < s)]
        new_matrix[i, j] = 0.5 * mat[m, n] + 0.5 * sum(mat[a, b] for a, b in clamped) / len(clamped)
    return new_matrix



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






if __name__ == '__main__':
    b = np.random.random((2**4,)*3)
    x = np.zeros(2**i)