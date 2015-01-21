from __future__ import print_function
import numpy

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Box(object):
    next_id = 0

    @staticmethod
    def get_next_id():
        Box.next_id += 1
        return Box.next_id

    def __init__(self, num_vectors, dim, ghost_zone_depth):
        """
        creates a box, based on create_box from level.c
        currently we are not worrying about alignment in python
        :param num_vectors:
        :param dim:
        :param ghost_zone_depth:
        :return:
        """
        self.global_box_id = Box.get_next_id()  # used to index into level.rank_of_box

        self.num_vectors = num_vectors
        self.dim = dim
        self.ghost_zone_depth = ghost_zone_depth

        self.j_stride = self.dim + 2 * self.ghost_zone_depth
        self.k_stride = self.j_stride * (self.dim + 2 * self.ghost_zone_depth)

        self.volume = (dim + 2 * self.ghost_zone_depth) * self.k_stride

        try:
            self.vectors = numpy.zeros([self.num_vectors]).astype(numpy.float64)
            self.vectors_base = numpy.zeros([self.num_vectors]).astype(numpy.float64)
        except Exception as exception:
            raise Exception("Box create failed, num_vectors={} dim={}, ghost_depth={}".format(
                self.num_vectors, self.dim, self.ghost_zone_depth))

    def add_vectors(self, new_vectors):
        if new_vectors < 1:
            return

        self.num_vectors += new_vectors
        try:
            self.vectors.resize([self.num_vectors])
        except Exception as exception:
            raise Exception("Box add_vectors failed, num_vectors={} dim={}, ghost_depth={} new_vectors={}".format(
                self.num_vectors, self.dim, self.ghost_zone_depth, new_vectors))




