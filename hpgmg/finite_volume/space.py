from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Coord(object):
    """
    implements a 3d coordinate, with convenience methods
    """
    def __init__(self, i=0, j=0, k=0):
        self.i, self.j, self.k = i, j, k

    def to_tuple(self):
        return self.i, self.j, self.k

    @staticmethod
    def from_tuple(tup):
        return Space(tup[0], tup[1], tup[2])

    def __mul__(self, other):
        assert isinstance(other, int)
        return Space(self.i * other, self.j * other, self.k * other)

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, Coord):
            return self.to_tuple() == other.to_tuple()
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Coord):
            return Coord(self.i + other.i, self.j + other.j, self.k + other.k)
        elif isinstance(other, tuple):
            if len(other) == 3:
                return Coord(self.i + other[0], self.j + other[1], self.k + other[2])
        elif isinstance(other, int):
            return Coord(self.i + other, self.j + other, self.k + other)
        else:
            raise Exception("Coord {}, can't add {}".format(self.to_tuple(), other))

    __radd__ = __add__

    def volume(self):
        """
        enclosed volume of space
        :return:
        """
        return self.i * self.j * self.k

    def foreach(self):
        """
        iterate overall all coordinates in the space
        :return:
        """
        for k in range(self.k):
            for j in range(self.j):
                for i in range(self.i):
                    yield Coord(i, j, k)

    def foreach_tiled(self, tiling_coord):
        """
        iterate overall all coordinates in the space
        :return:
        """
        for k in range(0, self.k, tiling_coord.k):
            for j in range(0, self.j, tiling_coord.j):
                for i in range(0, self.i, tiling_coord.i):
                    yield Coord(i, j, k)

    def foreach_index_1d(self):
        """
        iterate overall all coordinates in the space, as 1d array offset
        :return:
        """
        for k in range(self.k):
            for j in range(self.j):
                for i in range(self.i):
                    yield self.index_3d_to_1d((i, j, k))

    def index_3d_to_1d(self, coord):
        """
        convert the Coord coord to a scalar, using this shape as dimensional basis
        :param coord:
        :return: a scalar representing a 1d position to where coord would be in this linearized shape
        """
        if isinstance(coord, Coord):
            return (coord.i * self.j * self.k) + (coord.j * self.k) + coord.k
        elif isinstance(coord, (list, tuple)):
            return (coord[0] * self.j * self.k) + (coord[1] * self.k) + coord[2]
        else:
            raise Exception("Space({}).index_3d_to_1d bad argument {}".format(self.to_tuple(), coord))

    def index_1d_to_3d(self, index):
        assert isinstance(index, int)
        i = index // (self.j * self.k)
        j = (index - (i * self.j * self.k)) // self.k
        k = index % self.k
        return i, j, k

    def constrain_to_min(self, constraint):
        assert isinstance(constraint, Coord)
        return Coord(max(self.i, constraint.i), min(self.j, constraint.j), min(self.k, constraint.k))


class Space(Coord):
    pass



class Section(object):
    """
    implements a section of a 3d space with lower
    and upper bounds
    """
    def __init__(self, low, high):
        self.low, self.high = low, high
