from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from space import Coord


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
        self.data_ptr = data_ptr

    def copy(self, box_index=None, coord=None, i_stride=None, j_stride=None, scale=None, data_ptr=None):
        return GridCoordinate(
            self.box_index if box_index is None else box_index,
            self.coord if coord is None else coord,
            self.i_stride if i_stride is None else i_stride,
            self.j_stride if j_stride is None else j_stride,
            self.scale if scale is None else scale,
            self.data_ptr if data_ptr is None else data_ptr
        )


class BlockCopy(object):
    def __init__(self, subtype, dim, read_info, write_info):
        self.subtype = 0
        self.shape = dim
        self.read_info = read_info
        self.write_info = write_info


