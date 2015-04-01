from __future__ import print_function
import numpy as np

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


from hpgmg.finite_volume.space import Space, Coord


class Box(object):
    def __init__(self, level, coord, box_element_space):
        """
        creates a box, based on create_box from level.c
        currently we are not worrying about alignment in python
        :param coord: typically the 1-d index of box
        :param num_vectors:
        :param box_dim_size:
        :param ghost_zone_size:
        :return:
        """
        assert isinstance(coord, Coord)

        self.level = level
        self.coord = coord
        self.global_box_id = self.level.box_space.index_to_1d(self.coord)
        self.elements = np.empty(box_element_space, dtype=object)

    def add_vectors(self, new_vectors):
        if new_vectors < 1:
            return

        self.num_vectors += new_vectors
        try:
            self.vectors.resize([self.num_vectors])
        except Exception as exception:
            raise Exception("Box add_vectors failed, num_vectors={} box_dim_size={}, ghost_depth={} new_vectors={}".format(
                self.num_vectors, self.box_dim_size, self.ghost_zone_size, new_vectors))




