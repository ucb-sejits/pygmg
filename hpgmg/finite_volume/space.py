from __future__ import print_function, division
__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Coord(tuple):
    """
    implements a 3d coordinate, with convenience methods
    """
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (Coord, tuple)):
            return tuple.__new__(Coord, (args[0][0], args[0][1], args[0][2]))
        elif len(args) == 3:
            return tuple.__new__(Coord, (args[0], args[1], args[2]))
        else:
            raise Exception("Attempting to create a Coord from {}".format(*args))

    @property
    def i(self):
        return self[0]

    @property
    def j(self):
        return self[1]

    @property
    def k(self):
        return self[2]

    def to_tuple(self):
        return self.i, self.j, self.k

    def __mul__(self, other):
        assert isinstance(other, int)
        return Coord(self.i * other, self.j * other, self.k * other)

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, Coord):
            return Coord(self.i + other.i, self.j + other.j, self.k + other.k)
        elif isinstance(other, tuple):
            if len(other) == 3:
                return Coord(self.i + other[0], self.j + other[1], self.k + other[2])
        elif isinstance(other, int):
            return Coord(self.i + other, self.j + other, self.k + other)
        else:
            raise Exception("Coord {}, can't add {}".format(self, other))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Coord):
            return Coord(self.i - other.i, self.j - other.j, self.k - other.k)
        elif isinstance(other, tuple):
            if len(other) == 3:
                return Coord(self.i - other[0], self.j - other[1], self.k - other[2])
        elif isinstance(other, int):
            return Coord(self.i - other, self.j - other, self.k - other)
        else:
            raise Exception("Coord {}, can't add {}".format(self, other))

    def __rsub__(self, other):
        if isinstance(other, Coord):
            return Coord(other.i - self.i, other.j - self.j, other.k - self.k)
        elif isinstance(other, tuple):
            if len(other) == 3:
                return Coord(other[0] - self.i, other[1] - self.j, other[2] - self.k)
        elif isinstance(other, int):
            return Coord(other - self.i, other - self.j, other - self.k)
        else:
            raise Exception("Coord {}, can't add {}".format(self, other))

    def __floordiv__(self, other):
        if isinstance(other, Coord):
            return Coord(self.i // other.i, self.j // other.j, self.k // other.k)
        elif isinstance(other, tuple):
            if len(other) == 3:
                return Coord(self.i // other[0], self.j // other[1], self.k // other[2])
        elif isinstance(other, int):
            return Coord(self.i // other, self.j // other, self.k // other)
        else:
            raise Exception("Coord {}, can't floor divide {}".format(self, other))

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
            raise Exception("Space({}).index_3d_to_1d bad argument {}".format(self, coord))

    def index_1d_to_3d(self, index):
        assert isinstance(index, int)
        i = index // (self.j * self.k)
        j = (index - (i * self.j * self.k)) // self.k
        k = index % self.k
        return i, j, k

    def constrain_to_min(self, constraint):
        assert isinstance(constraint, Coord)
        return Coord(max(self.i, constraint.i), min(self.j, constraint.j), min(self.k, constraint.k))

    def is_space(self):
        return odd(self.i) and odd(self.j) and odd(self.k)

    def in_space(self, space):
        assert isinstance(space, Coord)
        return 0 <= self.i < space.i and 0 <= self.j < space.j and 0 <= self.k < space.k

    def halve(self):
        assert self.is_space()
        return (self // 2) + 1

    def double(self):
        assert self.is_space()
        return (self * 2) + -1

    RelativeFaceNeighborCoords = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    RelativeForwardFaceNeighborCoords = [
        c
        for c in RelativeFaceNeighborCoords
        if 0 <= c[0] and 0 <= c[1] and 0 <= c[2]
    ]

    RelativeEdgeNeighborCoords = [
        (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
        (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
        (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
    ]

    RelativeForwardEdgeNeighborCoords = [
        c
        for c in RelativeEdgeNeighborCoords
        if 0 <= c[0] and 0 <= c[1] and 0 <= c[2]
    ]

    RelativeCornerNeighborCoords = [
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ]

    def legal_face_neighbor_coords(self, space=None):
        for offset_tuple in Coord.RelativeFaceNeighborCoords:
            coord = self + offset_tuple
            if space is None:
                if 0 <= coord.i and 0 <= coord.j and 0 <= coord.k:
                    yield coord
            elif coord.in_space(space):
                yield coord

    def legal_edge_neighbor_coords(self, space=None):
        for offset_tuple in Coord.RelativeEdgeNeighborCoords:
            coord = self + offset_tuple
            if space is None:
                if 0 <= coord.i and 0 <= coord.j and 0 <= coord.k:
                    yield coord
            elif coord.in_space(space):
                yield coord

    def legal_corner_neighbor_coords(self, space=None):
        for offset_tuple in Coord.RelativeCornerNeighborCoords:
            coord = self + offset_tuple
            if space is None:
                if 0 <= coord.i and 0 <= coord.j and 0 <= coord.k:
                    yield coord
            elif coord.in_space(space):
                yield coord

    def legal_neighbor_coords(self, space=None):
        for coord in self.legal_face_neighbor_coords(space):
            yield coord
        for coord in self.legal_edge_neighbor_coords(space):
            yield coord
        for coord in self.legal_corner_neighbor_coords(space):
            yield coord

    def is_black(self):
        return (self.i + self.j + self.k) % 2 == 0

    def is_red(self):
        return not self.is_black()


def odd(x):
    return x % 2 == 1


class Space(Coord):
    pass

class Section(object):
    """
    implements a section of a 3d space with lower
    and upper bounds
    """
    def __init__(self, low, high):
        self.low, self.high = low, high
