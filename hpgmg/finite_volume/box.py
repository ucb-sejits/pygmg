from __future__ import print_function
import numpy

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


from hpgmg.finite_volume.shape import Shape


class Box(object):
    def __init__(self, global_id, num_vectors, box_dim_size, ghost_zone_size):
        """
        creates a box, based on create_box from level.c
        currently we are not worrying about alignment in python
        :param global_id: typically the 1-d index of box
        :param num_vectors:
        :param box_dim_size:
        :param ghost_zone_size:
        :return:
        """
        self.global_box_id = global_id
        self.num_vectors = num_vectors
        self.box_dim_size = box_dim_size
        self.ghost_zone_size = ghost_zone_size

        self.j_stride = self.box_dim_size + 2 * self.ghost_zone_size
        self.i_stride = self.j_stride * (self.box_dim_size + 2 * self.ghost_zone_size)

        self.volume = (box_dim_size + 2 * self.ghost_zone_size) * self.i_stride
        self.low = Shape()

        try:
            self.vectors = numpy.zeros([self.num_vectors]).astype(numpy.float64)
            self.vectors_base = numpy.zeros([self.num_vectors]).astype(numpy.float64)
        except Exception as exception:
            raise Exception("Box create failed, num_vectors={} box_dim_size={}, ghost_depth={}".format(
                self.num_vectors, self.box_dim_size, self.ghost_zone_size))

    def add_vectors(self, new_vectors):
        if new_vectors < 1:
            return

        self.num_vectors += new_vectors
        try:
            self.vectors.resize([self.num_vectors])
        except Exception as exception:
            raise Exception("Box add_vectors failed, num_vectors={} box_dim_size={}, ghost_depth={} new_vectors={}".format(
                self.num_vectors, self.box_dim_size, self.ghost_zone_size, new_vectors))




