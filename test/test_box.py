from __future__ import print_function
__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.level import Level
from hpgmg.finite_volume.box import Box
from hpgmg.finite_volume.space import Coord
from hpgmg.finite_volume.boundary_condition import BoundaryCondition


class TestBox(unittest.TestCase):
    def test_create(self):
        level = Level(64, 1, 1, 64, BoundaryCondition.DIRICHLET, 0, 1)
        box = Box(level, Coord(0, 0, 0), 20, 1, 1)
        self.assertTrue(box is not None)
        self.assertEqual(box.j_stride, 3)
        self.assertEqual(box.i_stride, 9)

        box.vectors[10] = 3.14

        box.add_vectors(10)
        self.assertEqual(box.vectors.size, 30)
        self.assertEqual(box.vectors[10], 3.14)