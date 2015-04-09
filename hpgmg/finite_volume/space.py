from __future__ import print_function, division

import collections
import numbers
import itertools
import numpy as np
try:
    # Python 2
    from itertools import izip_longest
except ImportError:
    # Python 3
    from itertools import zip_longest as izip_longest

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Vector(tuple):
    def __new__(cls, *args):
        if len(args) > 1:
            return tuple.__new__(cls, args)
        return tuple.__new__(cls, args[0])

    def __mul__(self, other):
        """scalar product if other is a number, otherwise does inner product"""
        if isinstance(other, numbers.Real):
            return type(self)(other*el for el in self)
        elif isinstance(other, collections.Iterable):
            return sum(i*j for i, j in izip_longest(self, other, fillvalue=0))
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        """adds a constant to each element if other is a number, otherwise does pairwise addition"""
        if isinstance(other, numbers.Real):
            return type(self)(other+el for el in self)
        elif isinstance(other, collections.Iterable):
            return type(self)(i+j for i, j in izip_longest(self, other, fillvalue=0))
        else:
            return NotImplemented

    __radd__ = __add__

    @property
    def i(self):
        return self[0]

    @property
    def j(self):
        return self[1]

    @property
    def k(self):
        return self[2]

    @property
    def ndim(self):
        return len(self)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __div__(self, other):
        if isinstance(other, numbers.Real):
            return self * (1/other)
        if isinstance(other, collections.Iterable):
            return type(self)(i/j for i, j in izip_longest(self, other, fillvalue=0))
        return NotImplemented

    __truediv__ = __div__

    def __floordiv__(self, other):
        if isinstance(other, numbers.Real):
            return type(self)(i//other for i in self)
        elif isinstance(other, collections.Iterable):
            return type(self)(i//j for i, j in izip_longest(self, other, fillvalue=0))
        return NotImplemented

    def in_space(self, space):
        assert isinstance(space, (Space, Coord, Vector)), "error space {} {}".format(type(space), space)
        return 0 <= self.i < space.i and 0 <= self.j < self.j and 0 <= self.k < space.k

    def to_1d(self, space):
        assert self.ndim == space.ndim
        return sum([x * y for x, y in zip(self, space.strides())])


class Coord(Vector):
    def __new__(cls, *args):
        args = args if len(args) > 1 else args[0]
        return Vector.__new__(cls, map(int, args))

    def __mul__(self, other):
        """
        element-wise multiplication
        :param other:
        :return:
        """
        if isinstance(other, numbers.Real):
            return Space(int(dim * other) for dim in self)
        if isinstance(other, collections.Iterable):
            return Space(dim * scale for dim, scale in izip_longest(self, other, fillvalue=0))
        return NotImplemented

    def __rmul__(self, other):
        """
        only multiplying a space by an int makes sense
        """
        if isinstance(other, int):
            return self * other
        return NotImplemented

    def __div__(self, other):
        if isinstance(other, numbers.Real):
            return self * (1/other)
        if isinstance(other, collections.Iterable):
            return type(self)(i/j for i, j in izip_longest(self, other, fillvalue=0))
        return NotImplemented

    def __sub__(self, other):
        return self + -other


class Space(Coord):
    """
    represents a multidimensional space, i.e. a 3x3x3 space would have indices
    0-2 in each dimension, created with Space(3,3,3).

    Multiplying a space multiplies the maximum index in each direction by that constant.
    a 3x3x3 space has indices 0-2 in each direction, so that space * 2 gives 0-4 in each direction,
    or a 5x5x5 space.
    """

    def __new__(cls, *args):
        if len(args) > 1:
            return tuple.__new__(cls, args)
        return tuple.__new__(cls, args[0])

    @property
    def volume(self):
        return np.multiply.reduce(self)

    @property
    def points(self):
        """Iterates over the points in the space"""
        return (Coord(coord) for coord in itertools.product(*[range(i) for i in self]))

    def interior_points(self, halo):
        """Iterates over the points in the space"""
        return (Coord(coord) for coord in itertools.product(*[range(g, i-g) for i, g in zip(self, halo)]))

    def __contains__(self, item):
        """Determines if the coordinate is in this space"""
        if isinstance(item, collections.Iterable):
            return all(0 <= i < bound for i, bound in izip_longest(item, self, fillvalue=0))
        return NotImplemented

    def neighbor_deltas(self):
        return itertools.product((-1, 0, 1), repeat=len(self))

    def neighbors(self, coord, norm=None):
        """
        Calculates the neighbors of the coordinate in n-space with 1-norm 'norm'.
        I.E. in 3-space, a 'corner' would have a 1-norm of 3, an 'edge' would have a
        1-norm of 2, and a 'face' would have a 1-norm of 1

        If norm is specified, then it filters for points with norm 'norm'
        """
        for delta in self.neighbor_deltas():
            if norm is None or np.linalg.norm(delta, 1) == norm:
                new_pt = coord + delta
                if new_pt in self:
                    yield new_pt

    def neighborhood_slice(self, coord):
        """
        Calculates the slice surrounding a coord, so self[slice] is its neighborhood
        """
        slices = []
        for vector, bound in itertools.izip_longest(coord, self, fillvalue=0):
            lower = max(0, vector - 1)
            upper = min(bound, vector + 2)
            slices.append(slice(lower, upper))
        return tuple(slices)

    def is_boundary_point(self, coord):
        return any(i == 0 or i == lim - 1 for i, lim in zip(coord, self))

    def strides(self):
        return [np.multiply.reduce(self[x:]) for x in range(1, len(self))] + [1]

    def index_to_1d(self, index):
        return sum([x*y for x, y in zip(index, self.strides())])

    def index_from_1d(self, index):
        def rem(value, strides):
            if len(strides) == 1:
                return [value]
            else:
                return [value // strides[0]] + rem(value % strides[0], strides[1:])

        return Coord(rem(index, self.strides()))


class Section(object):
    """
    implements a section of a 3d space with lower
    and upper bounds
    """
    def __init__(self, low, high):
        self.low, self.high = low, high
