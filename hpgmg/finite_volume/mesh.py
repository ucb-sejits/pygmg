from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np

from space import Coord, Space


class Mesh(np.ndarray):
    def __new__(cls, *args, **kwargs):
        obj = np.ndarray(*args, **kwargs).view(Mesh)
        obj.fill(0)
        return obj

    @property
    def space(self):
        return Space(self.shape)

    def indices(self):
        return self.space.points

    def assign_to_all(self, value):
        for index in self.indices():
            self[index] = value

    def __eq__(self, other):
        if self.space != other.space:
            return False
        else:
            for i in self.indices():
                if self[i] != other[i]:
                    return False
        return True

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
                    print("{:4.1f}".format(self[(i, j, k)]), end=" ")
                print()
            print()
        print()
