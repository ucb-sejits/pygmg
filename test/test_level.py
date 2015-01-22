from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.level import Level, BoundaryCondition, BC_DIRICHLET, BlockCounts


class TestLevel(unittest.TestCase):
    def test_create(self):
        my_rank = 0
        num_tasks = 1
        allocated = BlockCounts(100, 100)
        level = Level(8, 1 << 6, 1, 100, BC_DIRICHLET, my_rank=my_rank, num_ranks=2)
        self.assertTrue(level is not None)
