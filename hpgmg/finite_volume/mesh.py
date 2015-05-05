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

    def dump(self, message=None):
        """
        print this mesh, if 3d axes go up the page
        if 2d then standard over and down
        :return:
        """
        if message:
            print("==,MESHSTART,{},{}\n".format(message, self.space-2))

        if len(self.space) == 3:
            # block_list = [
            #     ((0, 0, 0), (8, 8, 16)),
            #     ((0, 8, 0), (8, 16, 16)),
            #     ((8, 0, 0), (16, 8, 16)),
            #     ((8, 8, 0), (16, 16, 16)),
            # ]
            # for ind, ends in enumerate(block_list):
            #     lo, hi = ends
            #     print("==block {} {} {}".format(ind, lo, hi))
            #     for ii in range(lo[0], hi[0]):
            #         i = ii + 1
            #         # print("i  {}".format(i))
            #         for jj in range(lo[1], hi[1]):
            #             j = jj + 1
            #             print("==", end='')
            #             for kk in range(lo[2], hi[2]):
            #                 k = kk + 1
            #                 print("{:f},".format(self[(i, j, k)]), end="")
            #             print()
            #         print("==")
            # if message:
            #     print("==MESHEND-{}-\n".format(message))
            max_i, max_j, max_k = self.shape

            print("==,block,0,1,1,1,{},{},{}".format(max_i-1, max_j-1, max_k-1))
            for i in range(1, max_i-1):
                # print("i  {}".format(i))
                for j in range(1, max_j-1):
                    print("==,{:02d},{:02d},".format(i-1, j-1), end='')
                    for k in range(1, max_k-1):
                        print("{:f},".format(self[(i, j, k)]), end="")
                    print()
            if message:
                print("==,MESHEND,{},\n".format(message))

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
