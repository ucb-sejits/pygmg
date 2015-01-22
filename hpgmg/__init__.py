__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from collections import namedtuple

Coord = namedtuple("Coord", ['i', 'j', 'k'])
CoordStride = namedtuple('CoordStride', ['coord', 'stride_j', 'stride_k'])