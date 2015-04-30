from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np

import hpgmg.finite_volume.space as space


class Mesh(np.ndarray):
    def __new__(cls, *args, **kwargs):
        obj = np.ndarray(*args, **kwargs).view(cls)
        obj.fill(0)
        return obj

    @property
    def space(self):
        return space.Space(self.shape)

    def indices(self):
        return self.space.points

    def assign_to_all(self, value):
        for index in self.indices():
            self[index] = value

    def __eq__(self, other):
        assert hasattr(other, 'space'), "mesh cannot test equality against {}".format(other)
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
        print this mesh, if 3d axes go up the page
        if 2d then standard over and down
        :return:
        """
        if message:
            print("Mesh print {} shape {}".format(message, self.shape))

        if len(self.space) == 3:
            max_i, max_j, max_k = self.shape

            for i in range(max_i-1, -1, -1):
                # print("i  {}".format(i))
                for j in range(max_j-1, -1, -1):
                    print(" "*j*2, end="")
                    for k in range(max_k):
                        print("{:10.2f}".format(self[(i, j, k)]), end=" ")
                    print()
                print()
            print()
        elif len(self.space) == 2:
            max_i, max_j = self.shape

            for i in range(max_i):
                # print("i  {}".format(i))
                for j in range(max_j):
                    print("{:10.5f}".format(self[(i, j)]), end=" ")
                print()
            print()
        else:
            print("I don't know how to mesh with {} dimensions".format(self.space.ndim))

    def zero(self):
        for index in self.indices():
            self[index] = 0.0
