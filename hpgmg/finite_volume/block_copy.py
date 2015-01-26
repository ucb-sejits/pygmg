from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg.finite_volume.space import Coord


class GridCoordinate(object):
    """
    information on where to extract data or where to insert data
    from level.h  nested struct in blockCopy_type
    """
    def __init__(self, box_index, coord, i_stride, j_stride, scale=1, data_ptr=None):
        assert isinstance(box_index, int)
        assert isinstance(coord, Coord)
        assert isinstance(i_stride, int)
        assert isinstance(j_stride, int)

        self.box_index = box_index
        self.coord = coord
        self.i_stride, self.j_stride = i_stride, j_stride


class BlockCopy(object):
    def __init__(self, shape):
        self.subtype = 0
        self.shape = shape
        self.read = GridCoordinate(Coord(0, 0, 0), 0, 0)
        self.write = GridCoordinate(Coord(0, 0, 0), 0, 0)


