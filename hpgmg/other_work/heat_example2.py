from __future__ import print_function
import stencil_code

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

"""
python version of matlab example from
http://en.wikipedia.org/wiki/Laplacian_matrix#Example_of_the_Operator_on_a_Grid

TODO: graphing not quite right
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


from stencil_code.stencil_kernel import Stencil, product
from stencil_code.neighborhood import Neighborhood


class Laplacian(Stencil):
    neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=1, dim=2, include_origin=False)]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = 0.5 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += 0.125 * in_grid[y]


class Average(Stencil):
    neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=1, dim=2, include_origin=True)]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = 0
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]
            out_grid[x] /= 5.0

#The number of pixels along a dimension of the image
N = 20
# The image
b = np.zeros([N, N])
#  The adjacency matrix

laplacian = Laplacian(backend='c', boundary_handling='clamp')
print("neighborhoods {}".format(laplacian.neighborhoods[0]))

# Initial condition (place a few large positive values around and
# make everything else zero)
data = np.zeros([N, N])
# data[2:5, 2:5] = 5
# data[10:15, 10:15] = 10
# data[2:5, 8:13] = 7
data[:, 4] = 30.0
data[4, :] = 30.0
data = data[:]


trials = 100
for ti in range(0, trials):
    t = ti / 0.5

    if ti % (trials // 5) == 0:
        plt.imshow(data, cmap='gray')
        plt.title("iteration {} avg {}".format(ti, np.average(data)))
        plt.show()

    new_data = laplacian(data)
    data = new_data

plt.imshow(data, cmap='gray')
plt.title("iteration {} avg {}".format(ti, np.average(data)))
plt.show()
