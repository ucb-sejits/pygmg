from __future__ import print_function
import numpy as np
from hpgmg.finite_volume.constants import Constants

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


from hpgmg.finite_volume.space import Coord


class Box(object):
    def __init__(self, level, coord):
        """
        creates a box, based on create_box from level.c
        currently we are not worrying about alignment in python
        :param level: parent level
        :param coord:
        :return:
        """
        assert isinstance(coord, Coord)

        self.level = level
        self.coord = coord
        self.global_box_id = self.level.box_space.index_to_1d(self.coord)
        self.vectors = [
            np.empty(level.box_element_space, dtype=np.float32)
            for _ in Constants.vector_list()
        ]

    def points(self):
        for index in self.level.box_element_space.points:
            yield index

    def interior_points(self):
        for index in self.level.box_element_space.interior_points(self.level.ghost_space):
            yield index


