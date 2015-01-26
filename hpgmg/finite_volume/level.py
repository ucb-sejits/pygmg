from __future__ import print_function
__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy
import math

from collections import namedtuple
from hpgmg.finite_volume.operators.stencil_27_pt import stencil_get_radius
from hpgmg.finite_volume.space import Space, Coord
from hpgmg.finite_volume.box import Box
from hpgmg.finite_volume.boundary_condition import BoundaryCondition
from hpgmg.finite_volume.block_copy import GridCoordinate


BLOCK_COPY_TILE_I = 10000
BLOCK_COPY_TILE_J = 8
BLOCK_COPY_TILE_K = 8

VECTORS_RESERVED = 12  # total number of grids and the starting location for any auxiliary bottom solver grids


DecomposeLex = True  # todo: this should come from global configuration
DecomposeBisectionSpecial = False  # todo: this should come from global configuration


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


class Level(object):
    """
    a collections of boxes and organizations of boxes in blocks
    level starts with a 3d array rank_of_box that is used by a decomposition routine
    to label blocks with a rank, the level will only allocate boxes where the rank matches
    the my_rank of the level
    """
    def __init__(self, boxes_in_i, box_dim_size, box_ghost_size, box_vectors,
                 domain_boundary_condition, my_rank, num_ranks):
        """
        create a level, initialize everything you can
        :param boxes_in_i: number of boxes on an axis, appears to be used for all axes
        :param box_dim_size:
        :param box_ghost_size:
        :param box_vectors:
        :param domain_boundary_condition: DIRICHLET or PERIODIC
        :param my_rank:
        :param num_ranks:
        :return:
        """
        if my_rank == 0:
            print("\nattempting to create a {:d}^3 level (with {} BC) ".format(
                boxes_in_i * box_dim_size,
                "Dirichlet" if domain_boundary_condition == BoundaryCondition.DIRICHLET else "Periodic"), end="")
            print("using a {:d}^3 grid of {:d}^3 boxes and {:d} tasks...\n".format(
                box_dim_size*boxes_in_i, boxes_in_i, box_dim_size, num_ranks))

        if box_ghost_size < stencil_get_radius():
            raise Exception("Level creation: ghosts {:d} must be >= stencil_get_radius() {:d}".format(
                box_ghost_size, stencil_get_radius()))

        self.box_dim_size = box_dim_size
        self.box_ghost_size = box_ghost_size
        self.box_vectors = box_vectors
        self.boxes_in = Space(boxes_in_i, boxes_in_i, boxes_in_i)
        self.dim = self.boxes_in * self.box_dim_size
        self.is_active = True
        self.my_rank = my_rank
        self.num_ranks = num_ranks
        self.boundary_condition = BoundaryCondition(domain_boundary_condition)
        self.alpha_is_zero = -1

        self.rank_of_box = self.build_rank_of_box()

        if DecomposeLex:
            self.decompose_level_lex(num_ranks)
        elif DecomposeBisectionSpecial:
            self.decompose_level_bisection_special(num_ranks)
        else:
            self.decompose_level_bisection(num_ranks)

        self.my_boxes = self.build_boxes()
        self.num_my_boxes = self.my_boxes.size

        self.blocks = numpy.empty(0, dtype=object)
        self.tag = math.log2(self.boxes_in.i)  # todo: this is probably wrong

        self.krylov_iterations = 0
        self.ca_krylov_formations_of_g = 0
        self.vcycles_from_this_level = 0

    def build_rank_of_box(self):
        try:
            rank_of_box = numpy.empty(self.boxes_in.to_tuple()).astype(numpy.int32)
            rank_of_box.fill(-1)
            return rank_of_box
        except Exception:
            raise Exception("Level create could not allocate rank_of_box")

    def build_boxes(self):
        num_my_boxes = 0
        for index in self.boxes_in.foreach():
            if self.rank_of_box[index.to_tuple()] == self.my_rank:
                num_my_boxes += 1

        try:
            my_boxes = numpy.empty(num_my_boxes, dtype=object)
        except Exception:
            raise("Level build_boxes failed trying to create my_boxes num={}".format(self.num_my_boxes))

        box_index = 0
        for index in self.boxes_in.foreach():
            index_1d = self.boxes_in.index_3d_to_1d(index)
            if self.rank_of_box[index.to_tuple()] == self.my_rank:
                box = Box(self, index, self.box_vectors, self.box_dim_size, self.box_ghost_size)
                box.low = index * self.box_dim_size

                my_boxes[box_index] = box
                box_index += 1

        return my_boxes

    def create_initial_blocks(self):
        for box in self.my_boxes:
            self.append_block_to_list(box.dim, )

    def decompose_level_lex(self, ranks):
        for index in self.boxes_in.foreach():
            index_1d = self.boxes_in.index_3d_to_1d(index)
            self.rank_of_box[index.to_tuple()] = (ranks * index_1d) / self.boxes_in.volume()

    def decompose_level_bisection_special(self, num_ranks):
        raise Exception("decompose_level_bisection_special not implemented. Level shape {}".format(self.shape.to_tuple()))

    def decompose_level_bisection(self, num_ranks):
        raise Exception("decompose_level_bisection not implemented. Level shape {}".format(self.shape.to_tuple()))

    def print_decomposition(self):
        """
        decomposition labels each possible box referenced in rank_of_box with a
        rank.  this shows the labeling in a table like form
        :return:
        """
        if self.my_rank != 0:
            return

        print()
        for i in range(self.boxes_in.i-1, 0, -1):
            for j in range(self.boxes_in.k-1, 0, -1):
                print(" "*j, end="")
                for k in range(self.boxes_in.k):
                    print("{:4d}".format(self.rank_of_box[(i, j, k)]), end="")
                print()
            print()
        print()

    def append_block_to_list(self, dim, read_info, write_info, block_copy_tile, subtype):
        """
        This increases the number of blockCopies in the ghost zone exchange and thereby increases the thread-level parallelism
        FIX... move from lexicographical ordering of tiles to recursive (e.g. z-mort)

        read_/write_scale are used to stride appropriately when read and write loop iterations spaces are different
        ghostZone:     read_scale=1, write_scale=1
        interpolation: read_scale=1, write_scale=2
        restriction:   read_scale=2, write_scale=1
        FIX... dim_i,j,k -> read_dim_i,j,k, write_dim_i,j,k

        :param dim:
        :param read_info:
        :param write_info:
        :param block_copy_tile:
        :param subtype:
        :return:
        """
        assert isinstance(dim, Coord)
        assert isinstance(read_info, GridCoordinate)
        assert isinstance(write_info, GridCoordinate)
        assert isinstance(block_copy_tile, Coord)

        new_blocks = []
        for tiled_dim in dim.foreach_tiled(block_copy_tile):
            dim_mod = (dim - tiled_dim).constrain_to_min(block_copy_tile)

            new_block = BlockCopies(
                subtype=subtype,
                dim=dim_mod,
                read_info=read_info.copy(coord=read_info.coord + (read_info.scale * tiled_dim)),
                write_info=write_info.copy(coord=write_info.coord + (write_info.scale * tiled_dim))
            )
            new_blocks.append(new_block)

        self.blocks = numpy.append(self.blocks, new_blocks)

    def neighbor_indices(self, box_index):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    yield box_index

    def build_boundary_conditions(self, just_faces):
        assert(BoundaryCondition.valid_index(just_faces))

        # these 3 are defaults for periodic
        self.boundary_condition.blocks[just_faces] = None
        self.boundary_condition.num_blocks[just_faces] = 0
        self.boundary_condition.allocated_blocks[just_faces] = 0
        if self.boundary_condition.condition_type == BoundaryCondition.PERIODIC:
            return

        for box_index, box in enumerate(self.my_boxes):
            for di, dj, dk in BoundaryCondition.foreach_neighbor_delta():
                neighbor_vector = BoundaryCondition.neighbor_vector(di, dj, dk)










