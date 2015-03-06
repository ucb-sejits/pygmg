from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

"""
python version of matlab example from
http://en.wikipedia.org/wiki/Laplacian_matrix#Example_of_the_Operator_on_a_Grid

TODO: graphing not quite right
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


#The number of pixels along a dimension of the image
N = 20
# The image
b = np.zeros([N, N])
#  The adjacency matrix
Adj = np.zeros([N*N, N*N])

# Use 8 neighbors, and fill in the adjacency matrix
dx = [-1, 0, 1, -1, 1, -1, 0, 1]
dy = [-1, -1, -1, 0, 0, 1, 1, 1]
for x in range(N):
    for y in range(N):
        index = x * N + y
        for ne in range(len(dx)):
            new_x = x + dx[ne]
            new_y = y + dy[ne]
            if 0 <= new_x < N and 0 <= new_y < N:
                index2 = new_x * N + new_y
                Adj[index, index2] = 1


# BELOW IS THE KEY CODE THAT COMPUTES THE SOLUTION TO THE DIFFERENTIAL
# EQUATION

Deg = np.diag(sum(Adj, 2))  # Compute the degree matrix
L = Deg - Adj  # Compute the laplacian matrix in terms of the degree and adjacency matrices
eigen_values, eigen_vectors = np.linalg.eig(L)  # Compute the eigenvalues/vectors of the laplacian matrix

# Initial condition (place a few large positive values around and
# make everything else zero)
C0 = np.zeros([N, N])
C0[2:5, 2:5] = 5
C0[10:15, 10:15] = 10
C0[2:5, 8:13] = 7
C0 = C0[:]
#
print("eigen_vector shape {}".format(eigen_vectors.shape))
print("c0 shape {}".format(C0.shape))
C0V = eigen_vectors.T.dot(C0.flatten())  # Transform the initial condition into the coordinate system
# of the eigenvectors
for ti in range(0, 20):
    t = ti / 0.5
    # Loop through times and decay each initial component
    Phi = C0V.dot(np.exp(-eigen_values*t))  # Exponential decay for each component
    Phi = eigen_vectors * Phi  # Transform from eigenvector coordinate system to original coordinate system
    print("Phi.shape {} average {}".format(Phi.shape, np.average(Phi)))

    if ti % 5 == 4:
        plt.imshow(Phi, cmap='gray')
        plt.title("iteration {} avg {}".format(ti, np.average(Phi)))
        plt.show()
