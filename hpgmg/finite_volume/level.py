from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg import Coord, CoordStride
from collections import namedtuple
import numpy

BC_PERIODIC = 0
BC_DIRICHLET = 1

BLOCK_COPY_TILE_I = 10000
BLOCK_COPY_TILE_J = 8
BLOCK_COPY_TILE_K = 8


class BlockCopy(object):
    def __init__(self):
        self.subtype = 0
        self.dim = Coord(0, 0, 0)
        self.read = CoordStride(Coord(0, 0, 0), 0, 0)
        self.write = CoordStride(Coord(0, 0, 0), 0, 0)


class Communicator(object):
    """
    communicator_type in original has lot's of MPI specific code, let's see how
    far we can go without it
    """
    def __init__(self):
        self.num_receives = 0  # number of neighbors by type
        self.num_sends = 0  # number of neighbors by type

    def print(self, rank, level, print_send=True, print_receive=True, show_blocks=True):
        print("rank={:2d} level={:2d}".format(rank, level))
        if print_send:
            print("num_sends={:2d}".format(self.num_sends))
        if show_blocks:
            print("Todo: blocks not part of CommunicatorType yet")
        if print_receive:
            print("num_sends={:2d}".format(self.num_sends))


GZ = namedtuple('GZ', [
    'send_rank', 'send_box_id', 'send_box', 'send_dir', 'receive_rank', 'receive_box_id', 'receive_box'
])


BlockCounts = namedtuple('BlockCounts', ['all', 'just_faces'])
BlockCopies = namedtuple('BlockCopies', ['all', 'just_faces'])


class BoundaryCondition(object):
    def __init__(self, condition_type, allocated_blocks, num_blocks, blocks):
        assert isinstance(allocated_blocks, BlockCounts)
        assert isinstance(num_blocks, BlockCounts)
        assert isinstance(num_blocks, BlockCounts)

        self.allocated_blocks = allocated_blocks
        self.num_blocks = num_blocks
        self.blocks = blocks


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

