from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.level import Level, BoundaryCondition, BC_DIRICHLET, BlockCounts


class TestLevel(unittest.TestCase):
    def test_create(self):
        allocated = BlockCounts(100, 100)
        boundary_condition = BoundaryCondition(BC_DIRICHLET, allocated, allocated, allocated)
        level = Level(8, 1 << 6, 1, 100, boundary_condition, 1, 2)
        self.assertTrue(level is not None)
