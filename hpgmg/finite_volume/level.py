from __future__ import print_function
from hpgmg.finite_volume.element import Element

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
    def __init__(self, element_space, box_space, ghost_space,
                 domain_boundary_condition, my_rank=0, num_ranks=1):
        """
        create a level, initialize everything you can
        :param element_space: number of boxes on an axis, appears to be used for all axes
        :param box_dim_size:
        :param box_ghost_size:
        :param box_vectors:
        :param domain_boundary_condition: DIRICHLET or PERIODIC
        :param my_rank:
        :param num_ranks:
        :return:
        """
        assert(isinstance(element_space, Space))
        assert(isinstance(box_space, Space))
        assert(isinstance(ghost_space, Space))

        for size in ghost_space:
            if size < stencil_get_radius():
                raise Exception("Level creation: ghosts {:d} must be >= stencil_get_radius() {:d}".format(
                    ghost_space, stencil_get_radius()))

        if my_rank == 0:
            print("\nCreating level, {} elements, {} boxes, with {} boundary ".format(
                element_space, box_space,
                "Dirichlet" if domain_boundary_condition == BoundaryCondition.DIRICHLET else "Periodic"), end="")

        self.element_space = element_space
        self.box_space = box_space
        self.ghost_space = ghost_space
        self.box_element_space = Space(
            [elements/boxes for elements, boxes in zip(element_space, box_space)]
        )

        self.is_active = True
        self.my_rank = my_rank
        self.num_ranks = num_ranks
        self.boundary_condition = BoundaryCondition(domain_boundary_condition)
        self.alpha_is_zero = -1
        self.h = 0.0

        self.rank_of_box = numpy.zeros(self.box_space)
        self.my_boxes = []

        if DecomposeLex:
            self.decompose_level_lex(num_ranks)
        elif DecomposeBisectionSpecial:
            self.decompose_level_bisection_special(num_ranks)
        else:
            self.decompose_level_bisection(num_ranks)

        self.build_boxes()

        self.blocks = numpy.empty(0, dtype=object)  # empty gives us None
        self.tag = math.log2(self.box_space.i)  # todo: this is probably wrong

        self.krylov_iterations = 0
        self.ca_krylov_formations_of_g = 0
        self.vcycles_from_this_level = 0

    def build_boxes(self):
        """
        run throw the rank_of_box table and find entries that match this threads rank,
        build a box for each one of these
        :return:
        """
        for index in self.box_space.points:
            if self.rank_of_box[index] == self.my_rank:
                self.my_boxes.append(Box(self, index, self.box_element_space))

    def create_elements(self):
        for box in self.my_boxes:
            for element_index in self.box_element_space.points:
                box.elements[element_index] = Element()


    def decompose_level_lex(self, ranks):
        """
        simple lexicographical decomposition of the domain (i-j-k ordering)
        :param ranks:
        :return:
        """
        for index in self.box_space.points:
            index_1d = self.box_space.index_to_1d(index)
            self.rank_of_box[index] = (ranks * index_1d) / self.box_space.volume

    def decompose_level_bisection_special(self, num_ranks):
        raise Exception("decompose_level_bisection_special not implemented. Level shape {}".format(self.shape))

    def decompose_level_bisection(self, num_ranks):
        raise Exception("decompose_level_bisection not implemented. Level shape {}".format(self.shape))

    def print_decomposition(self):
        """
        decomposition labels each possible box referenced in rank_of_box with a
        rank.  this shows the labeling in a table like form
        :return:
        """
        if self.my_rank != 0:
            return

        print()
        for i in range(self.box_shape.i-1, 0, -1):
            for j in range(self.box_shape.k-1, 0, -1):
                print(" "*j, end="")
                for k in range(self.box_shape.k):
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
                my_box = box // self.box_dim_size
                neighbor_box = my_box + (di, dj ,dk)












