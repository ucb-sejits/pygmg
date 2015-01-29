from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np

from hpgmg.finite_volume.space import Coord


class Mesh(np.ndarray):
    def space(self):
        return Coord.from_tuple(self.shape)

    def __getitem__(self, item):
        if isinstance(item, Coord):
            return super(Mesh, self).__getitem__(item.to_tuple())
        else:
            return super(Mesh, self).__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, Coord):
            return super(Mesh, self).__setitem__(key.to_tuple(), value)
        else:
            return super(Mesh, self).__getitem__(key, value)

    def indices(self):
        for coord in self.space().foreach():
            yield coord

    @staticmethod
    def from_coord(coord):
        return Mesh(coord.to_tuple())