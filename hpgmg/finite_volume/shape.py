from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Shape(object):
    """
    implements a 3d coordinate
    """
    # todo: figure out c hpgmg striding, seems to consider k, least rapidly varying index
    def __init__(self, i=0, j=0, k=0):
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
        return (index[0] * self.j * self.k) + (index[1] * self.k) + index[2]

    def index_1d_to_3d(self, index):
        i = index // (self.j * self.k)
        j = (index - (i * self.j * self.k)) // self.k
        k = index % self.k
        return i, j, k

    def __mul__(self, other):
        assert isinstance(other, int)
        return Shape(self.i * other, self.j * other, self.k * other)

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, Shape):
            return self.tup() == other.tup()
        else:
            return False

    @staticmethod
    def from_tuple(tup):
        return Shape(tup[0], tup[1], tup[2])


class Section(object):
    """
    implements a section of a 3d space with lower
    and upper bounds
    """
    def __init__(self, low, high):
        self.low, self.high = low, high
