from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np

from hpgmg.finite_volume.space import Coord


class Mesh(np.ndarray):
    def space(self):
        return Coord(self.shape)

    def __getitem__(self, item):
        if isinstance(item, Coord):
            return super(Mesh, self).__getitem__(item)
        else:
            return super(Mesh, self).__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, Coord):
            return super(Mesh, self).__setitem__(key, value)
        else:
            return super(Mesh, self).__setitem__(key, value)

    def indices(self):
        for coord in self.space().foreach():
            yield coord

    def red_indices(self):
        """
        matrix origin we define to be black
        :return:
        """
        for index in self.indices():
            if index.is_red():
                yield index

    def black_indices(self):
        """
        matrix origin we define to be black
        :return:
        """
        for index in self.indices():
            if index.is_red():
                yield index

    @staticmethod
    def from_coord(coord):
        return Mesh(coord)

    def __contains__(self, item):
        if isinstance(item, Coord):
            item = item
        return all(
            0 <= index < max_dim for index, max_dim in zip(item, self.shape)
        )

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
