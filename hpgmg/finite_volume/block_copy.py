from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg import Coord, CoordStride


class BlockCopy(object):
    def __init__(self, shape):
        self.subtype = 0
        self.shape = shape
        self.read = CoordStride(Coord(0, 0, 0), 0, 0)
        self.write = CoordStride(Coord(0, 0, 0), 0, 0)


