from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy
import math

from collections import namedtuple
from hpgmg.finite_volume.operators.stencil_27_pt import stencil_get_radius
from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.box import Box
from hpgmg.finite_volume.boundary_condition import BoundaryCondition

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

    BOX_ALIGNMENTS = Vector(1, 2, 8)
    BLOCK_COPY_TILE = Vector(10000, 8, 8)

    def __init__(self, element_space, box_space, ghost_space,
                 domain_boundary_condition, my_rank=0, num_ranks=1):
        """
        create a level, initialize everything you can
        :param element_space: total number of finite vectors (cells) as vector
        :param box_space: total number of boxes along each dimension vector
        :param ghost_space: total number of ghosts as vector
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
            print("\nCreating level, {} vectors, {} boxes, with {} boundary ".format(
                element_space, box_space,
                "Dirichlet" if domain_boundary_condition == BoundaryCondition.DIRICHLET else "Periodic"))

        self.dimensions = len(element_space)
        self.element_space = element_space
        self.box_space = box_space
        self.ghost_space = ghost_space
        self.box_element_space = self.create_compute_box_size()

        self.is_active = True
        self.my_rank = my_rank
        self.num_ranks = num_ranks
        self.boundary_condition = BoundaryCondition(domain_boundary_condition)
        self.alpha_is_zero = None
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
                self.my_boxes.append(Box(self, index))

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
        raise Exception("decompose_level_bisection_special not implemented. Level shape {}".format(self.element_space))

    def decompose_level_bisection(self, num_ranks):
        raise Exception("decompose_level_bisection not implemented. Level shape {}".format(self.element_space))

    def print_decomposition(self):
        """
        decomposition labels each possible box referenced in rank_of_box with a
        rank.  this shows the labeling in a table like form
        :return:
        """
        if self.my_rank != 0:
            return

        print()
        for i in range(self.box_space.i-1, 0, -1):
            for j in range(self.box_space.k-1, 0, -1):
                print(" "*j, end="")
                for k in range(self.box_space.k):
                    print("{:4d}".format(self.rank_of_box[(i, j, k)]), end="")
                print()
            print()
        print()

    def create_compute_box_size(self):
        """
        based on total elements and number of boxes in each dimension
        compute the size, add in the ghost zones then align as necessary
        :return:
        """
        def compute_best_size_for(dim):
            size = ((self.element_space[dim]-1)//self.box_space[dim]) + 1
            size += 2 * self.ghost_space[dim]
            while size % Level.BOX_ALIGNMENTS[dim]:
                size += 1
            return size

        return Space([compute_best_size_for(dim) for dim in range(self.dimensions)])
