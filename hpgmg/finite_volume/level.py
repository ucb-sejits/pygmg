__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg import Coord
import numpy

BC_PERIODIC = 0
BC_DIRICHLET = 1

BLOCK_COPY_TILE_I = 10000
BLOCK_COPY_TILE_J = 8
BLOCK_COPY_TILE_K = 8



class CommunicatorType(object):
    """
    communicator_type in original has lot's of MPI specific code, let's see how
    far we can go without it
    """
    def __init__(self):
        self.num_receives = 0  # number of neighbors by type
        self.num_sends = 0  # number of neighbors by type


class BoxType(object):
    next_id = 0

    @staticmethod
    def get_next_id():
        BoxType.next_id += 1
        return BoxType.next_id

    def __init__(self, coord, ghost_zone_depth, vectors):
        assert isinstance(coord, Coord)
        assert isinstance(vectors, numpy.ndarray)

        self.global_box_id = BoxType.get_next_id()  # used to index into level.rank_of_box
        self.low = coord  # global coordinates of first non-ghost element of subdomain
        self.ghost_zone_depth = ghost_zone_depth
        self.vectors = vectors