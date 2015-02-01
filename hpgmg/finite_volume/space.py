from __future__ import print_function, division

import collections
import numbers
import itertools
import numpy as np

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

class Vector(tuple):
    def __new__(cls, *args):
        if len(args) > 1:
            return tuple.__new__(cls, args)
        return tuple.__new__(cls, args[0])

    def __mul__(self, other):
        """scalar product if other is a number, otherwise does inner product"""
        if isinstance(other, numbers.Number):
            return type(self)(other*el for el in self)
        elif isinstance(other, collections.Iterable):
            return sum(i*j for i,j in itertools.izip_longest(self, other))
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other):
        """adds a constant to each element if other is a number, otherwise does pairwise addition"""
        if isinstance(other, numbers.Number):
            return type(self)(other+el for el in self)
        elif isinstance(other, collections.Iterable):
            return type(self)(i+j for i,j in itertools.izip_longest(self, other))
        else:
            return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __div__(self, other):
        if isinstance(other, numbers.Number):
            return self * 1/other
        if isinstance(other, collections.Iterable):
            return type(self)(i/j for i,j in itertools.izip_longest(self, other))
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, numbers.Number):
            return type(self)(i//other for i in self)
        elif isinstance(other, collections.Iterable):
            return type(self)(i//j for i,j in itertools.izip_longest(self, other))
        return NotImplemented


class Coord(Vector):
    pass

class Space(tuple):
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

    def to_tuple(self):
        """legacy support"""
        return self

    @property
    def ndim(self):
        return len(self)

    @property
    def volume(self):
        return np.multiply.reduce(self)

    @property
    def points(self):
        """Iterates over the points in the space"""
        return (Coord(coord) for coord in itertools.product(*[range(i) for i in self]))

    def __contains__(self, item):
        """Determines if the coordinate is in this space"""
        if isinstance(item, collections.Iterable):
            return all(0 <= i < bound for i, bound in itertools.izip_longest(item, self, fillvalue=0))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, int):
            return Space((dim - 1) * other + 1 for dim in self)
        if isinstance(other, collections.Iterable):
            return Space((dim - 1) * scale + 1 for dim, scale in itertools.izip_longest(self, other, fillvalue=0))
        return NotImplemented

    def __rmul__(self, other):
        """
        only multiplying a space by an int makes sense
        """
        if isinstance(other, int):
            return self * other
        return NotImplemented

    def __div__(self, other):
        """
        performs integer division only. Can't have decimal dimensions
        """
        if isinstance(other, int):
            return Space((dim - 1)//other + 1 for dim in self)
        if isinstance(other, collections.Iterable):
            return Space((dim - 1) // scale for dim, scale in itertools.izip_longest(self, other, fillvalue=0))
        return NotImplemented

    __truediv__ = __floordiv__ = __div__

    def __add__(self, other):
        if isinstance(other, int):
            return Space(i + other for i in self)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self + -other

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


class Section(object):
    """
    implements a section of a 3d space with lower
    and upper bounds
    """
    def __init__(self, low, high):
        self.low, self.high = low, high
