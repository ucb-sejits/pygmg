from __future__ import print_function
from hpgmg.finite_volume.operators.boundary_conditions_fv_2 import BoundaryUpdaterV2
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.other_work.boundary_condition import BoundaryCondition


class TestBoundaryConditionV2(unittest.TestCase):
    """
    computes a ghost zone value based on quadratic interpolation
    :.......:.......|.......:.......:.......:.
    :   -   :   ?   | -5/2  :  1/2  :   0   :
    :.......:.......|.......:.......:.......:.

    """
    def test_boundary_updater_v2_dirichlet_2d(self):
        simple_solver = SimpleMultigridSolver.get_solver(["2", "-d", "2"])
        bu = BoundaryUpdaterV2(simple_solver)
        self.assertTrue(bu.apply == bu.apply_dirichlet)

        mesh = simple_solver.fine_level.cell_values

        for i in simple_solver.fine_level.interior_points():
            mesh[i] = 1

        # mesh.print("before bu")

        bu.apply(simple_solver.fine_level, mesh)

        mesh.print("after bu")

        self.assertEqual(mesh[(0, 0)], 4.0)  # corner is -4
        self.assertEqual(mesh[(0, 5)], 4.0)  # corner is -4
        self.assertEqual(mesh[(5, 0)], 4.0)  # corner is -4
        self.assertEqual(mesh[(5, 5)], 4.0)  # corner is -4

        self.assertEqual(mesh[(0, 1)], -2.0)  # edge is -2
        self.assertEqual(mesh[(0, 2)], -2.0)  # edge is -2
        self.assertEqual(mesh[(0, 3)], -2.0)  # edge is -2
        self.assertEqual(mesh[(0, 4)], -2.0)  # edge is -2

        self.assertEqual(mesh[(1, 0)], -2.0)  # edge is -2
        self.assertEqual(mesh[(2, 0)], -2.0)  # edge is -2
        self.assertEqual(mesh[(3, 0)], -2.0)  # edge is -2
        self.assertEqual(mesh[(4, 0)], -2.0)  # edge is -2

        self.assertEqual(mesh[(5, 1)], -2.0)  # edge is -2
        self.assertEqual(mesh[(5, 2)], -2.0)  # edge is -2
        self.assertEqual(mesh[(5, 3)], -2.0)  # edge is -2
        self.assertEqual(mesh[(5, 4)], -2.0)  # edge is -2

        self.assertEqual(mesh[(1, 5)], -2.0)  # edge is -2
        self.assertEqual(mesh[(2, 5)], -2.0)  # edge is -2
        self.assertEqual(mesh[(3, 5)], -2.0)  # edge is -2
        self.assertEqual(mesh[(4, 5)], -2.0)  # edge is -2

    def test_boundary_updater_v2_dirichlet_3d(self):
        simple_solver = SimpleMultigridSolver.get_solver(["2", "-d", "3"])
        bu = BoundaryUpdaterV2(simple_solver)
        self.assertTrue(bu.apply == bu.apply_dirichlet)

        mesh = simple_solver.fine_level.cell_values

        for i in simple_solver.fine_level.interior_points():
            mesh[i] = 1

        # mesh.print("before bu")

        bu.apply(simple_solver.fine_level, mesh)

        mesh.print("after bu")

        self.assertEqual(mesh[(5, 5, 5)], -8)  # top corner is -8
        self.assertEqual(mesh[(5, 0, 0)], -8)  # top corner is -8
        self.assertEqual(mesh[(5, 0, 5)], -8)  # top corner is -8
        self.assertEqual(mesh[(0, 0, 0)], -8)  # bottom corner is -8
        self.assertEqual(mesh[(0, 0, 5)], -8)  # bottom corner is -8
        self.assertEqual(mesh[(0, 5, 0)], -8)  # bottom corner is -8

        self.assertEqual(mesh[(5, 5, 1)], 4)  # edges are positive
        self.assertEqual(mesh[(5, 5, 2)], 4)  # edges are positive
        self.assertEqual(mesh[(5, 5, 3)], 4)  # edges are positive
        self.assertEqual(mesh[(5, 5, 4)], 4)  # edges are positive

        self.assertEqual(mesh[(0, 0, 1)], 4)  # edges are positive
        self.assertEqual(mesh[(0, 0, 2)], 4)  # edges are positive
        self.assertEqual(mesh[(0, 0, 3)], 4)  # edges are positive
        self.assertEqual(mesh[(0, 0, 4)], 4)  # edges are positive

        self.assertEqual(mesh[(0, 1, 1)], -2)  # faces are -2
        self.assertEqual(mesh[(0, 1, 2)], -2)  # faces are -2
        self.assertEqual(mesh[(0, 2, 3)], -2)  # faces are -2
        self.assertEqual(mesh[(0, 4, 4)], -2)  # faces are -2