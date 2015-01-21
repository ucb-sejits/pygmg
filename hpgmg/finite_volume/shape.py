from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Shape(object):
    """
    implements a 3d coordinate
    """
    # todo: figure out c hpgmg striding, seems to consider k, least rapidly varying index
    def __init__(self, i, j, k):
        self.i, self.j, self.k = i, j, k

    def tup(self):
        return self.i, self.j, self.k

    def volume(self):
        return self.i * self.j * self.k

    def foreach(self):
        for k in range(self.k):
            for j in range(self.j):
                for i in range(self.i):
                    yield i, j, k

    def index_3d_to_1d(self, index):
        return (index[2] * self.i * self.j) + (index[1] * self.i) + index[0]


class Section(object):
    """
    implements a section of a 3d space with lower
    and upper bounds
    """
    def __init__(self, low, high):
        self.low, self.high = low, high
