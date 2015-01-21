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


class Box(object):
    next_id = 0

    @staticmethod
    def get_next_id():
        Box.next_id += 1
        return BoxType.next_id

    def __init__(self, coord, ghost_zone_depth, vectors):
        assert isinstance(coord, Coord)
        assert isinstance(vectors, numpy.ndarray)

        self.global_box_id = Box.get_next_id()  # used to index into level.rank_of_box
        self.low = coord  # global coordinates of first non-ghost element of subdomain
        self.ghost_zone_depth = ghost_zone_depth
        self.vectors = vectors


class Level(object):
    def __init__(self, grid_spacing, box_dim, box_ghost_zone, num_box_vectors, boxes_in_ijk):
        self.grid_spacing = grid_spacing
        self.is_active = True
        self.box_dim = box_dim
        self.box_ghost_zone = box_ghost_zone

        assert isinstance(boxes_in_ijk, Coord)
        self.num_box_vectors = num_box_vectors
        self.dim = Coord(boxes_in_ijk.i * box_dim, boxes_in_ijk.j * box_dim, boxes_in_ijk.k * box_dim)

        self.boxes_in_ijk = boxes_in_ijk

        self.krylov_iterations = 0
        self.ca_krylov_formations_of_g = 0
        self.vcycles_from_this_level = 0

