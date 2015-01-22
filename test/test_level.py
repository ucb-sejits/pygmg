from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.level import Level, VECTORS_RESERVED, BC_DIRICHLET, BlockCounts
from hpgmg.finite_volume.operators.stencil_27_pt import stencil_get_radius

class TestLevel(unittest.TestCase):
    def test_create(self):
        my_rank = 0
        num_ranks = 1
        ghost_zone_size = stencil_get_radius()
        level = Level(8, 1 << 6, 1, VECTORS_RESERVED, BC_DIRICHLET, my_rank=my_rank, num_ranks=2)
        self.assertTrue(level is not None)
