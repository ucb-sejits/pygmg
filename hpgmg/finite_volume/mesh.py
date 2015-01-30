from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np

from hpgmg.finite_volume.space import Coord, Space


class Mesh(np.ndarray):

    @property
    def space(self):
        return Space(self.shape)

    def indices(self):
        return self.space.points

    def __contains__(self, item):
        return item in self.space

    def print(self, message=None):
        """
        decomposition labels each possible box referenced in rank_of_box with a
        rank.  this shows the labeling in a table like form
        :return:
        """
        max_i, max_j, max_k = self.shape

        if message:
            print("Mesh print {} shape {}".format(message, self.shape))

        for i in range(max_i-1, -1, -1):
            # print("i  {}".format(i))
            for j in range(max_j-1, -1, -1):
                print(" "*j, end="")
                for k in range(max_k):
                    print("{:4.0f}".format(self[(i, j, k)]), end="")
                print()
            print()
        print()
