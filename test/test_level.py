from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.level import Level, VECTORS_RESERVED
from hpgmg.finite_volume.boundary_condition import BoundaryCondition
from hpgmg.finite_volume.operators.stencil_27_pt import stencil_get_radius


class TestLevel(unittest.TestCase):
    def test_create(self):
        my_rank = 0
        num_ranks = 1
        ghost_zone_size = stencil_get_radius()
        level = Level(
            boxes_in_i=8, box_dim_size=1 << 4, box_ghost_size=1,
            box_vectors=VECTORS_RESERVED,
            domain_boundary_condition=BoundaryCondition.DIRICHLET,
            my_rank=my_rank, num_ranks=2
        )

        self.assertTrue(level is not None)

        # level.print_decomposition()

        self.assertEqual(level.rank_of_box.size, 8**3)
        self.assertEqual(level.my_boxes.size, (8**3)//2)  # actual boxes decomposed

