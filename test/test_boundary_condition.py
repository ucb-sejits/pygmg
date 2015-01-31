from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.boundary_condition import BoundaryCondition


class TestBoundaryCondition(unittest.TestCase):
    def test_basics(self):
        BC = BoundaryCondition
        self.assertEqual(len(BC.Faces), 27)
        self.assertEqual(len(BC.Edges), 27)
        self.assertEqual(len(BC.Corners), 27)

        self.assertTrue(BC.is_corner(BC.neighbor_vector(-1, -1, -1)))
        self.assertTrue(BC.is_corner(BC.neighbor_vector(1, 1, 1)))
        self.assertTrue(BC.is_corner(BC.neighbor_vector(-1, 1, -1)))

        self.assertTrue(BC.is_edge(BC.neighbor_vector(-1, 0, -1)))
        self.assertTrue(BC.is_edge(BC.neighbor_vector(-1, 0, 1)))
        self.assertTrue(BC.is_edge(BC.neighbor_vector(1, 0, -1)))

        self.assertTrue(BC.is_face(BC.neighbor_vector(-1, 0, 0)))
        self.assertTrue(BC.is_face(BC.neighbor_vector(0, 0, 1)))
        self.assertTrue(BC.is_face(BC.neighbor_vector(1, 0, 0)))

    def test_iteration(self):
        BC = BoundaryCondition
        self.assertEqual(len(list(BC.foreach_neighbor_delta())), 27)

