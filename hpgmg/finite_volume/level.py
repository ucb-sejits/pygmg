from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy
import math

from hpgmg import Coord, CoordStride
from collections import namedtuple
from hpgmg.finite_volume.operators.stencil_27_pt import stencil_get_radius

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
    def __init__(self, boxes_in_i, box_dim, box_ghosts, box_vectors, domain_boundary_condition, my_rank, num_ranks):
        total_boxes = boxes_in_i ** 3

        if my_rank == 0:
            print("\nattempting to create a %d^3 level (with {} BC)".format(
                "Dirichlet" if domain_boundary_condition == BC_DIRICHLET else "Periodic"), end="")
            print("using a {:d}^3 grid of {:d}^3 boxes and {:d} tasks...\n".format(
                box_dim*boxes_in_i, boxes_in_i, box_dim, num_ranks))

        if box_ghosts < stencil_get_radius():
            raise Exception("Level creation: ghosts {:d} must be >= stencil_get_radius() {:d}".format(
                box_ghosts, stencil_get_radius()))

        self.is_active = True
        self.box_dim = box_dim
        self.box_ghost_zone = box_ghosts
        self.box_vectors = box_vectors
        self.boxes_in = Coord(boxes_in_i, boxes_in_i, boxes_in_i)
        self.dim = Coord(box_dim * self.boxes_in.i, box_dim * self.boxes_in.j, box_dim * self.boxes_in.k)
        self.my_rank = my_rank
        self.num_ranks = num_ranks
        self.boundary_conditions = domain_boundary_condition
        self.alpha_is_zero = -1
        self.my_blocks = None
        self.num_my_blocks = 0
        self.allocated_blocks = 0
        self.tag = math.log2(self.dim.i)

        try:
            self.rank_of_box = numpy.zeros([self.boxes_in.i * self.boxes_in.j * self.boxes_in.j]).astype(numpy.int32)
        except Exception as e:
            raise Exception("Level create could not allocate rank_of_box")

        self.krylov_iterations = 0
        self.ca_krylov_formations_of_g = 0
        self.vcycles_from_this_level = 0

