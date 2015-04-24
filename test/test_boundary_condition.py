from __future__ import print_function
from hpgmg.finite_volume.operators.boundary_conditions_fv import BoundaryUpdaterV1
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.boundary_condition import BoundaryCondition


class TestBoundaryCondition(unittest.TestCase):
    def test_basics(self):
        bc = BoundaryCondition
        self.assertEqual(len(bc.Faces), 27)
        self.assertEqual(len(bc.Edges), 27)
        self.assertEqual(len(bc.Corners), 27)

        self.assertTrue(bc.is_corner(bc.neighbor_vector(-1, -1, -1)))
        self.assertTrue(bc.is_corner(bc.neighbor_vector(1, 1, 1)))
        self.assertTrue(bc.is_corner(bc.neighbor_vector(-1, 1, -1)))

        self.assertTrue(bc.is_edge(bc.neighbor_vector(-1, 0, -1)))
        self.assertTrue(bc.is_edge(bc.neighbor_vector(-1, 0, 1)))
        self.assertTrue(bc.is_edge(bc.neighbor_vector(1, 0, -1)))

        self.assertTrue(bc.is_face(bc.neighbor_vector(-1, 0, 0)))
        self.assertTrue(bc.is_face(bc.neighbor_vector(0, 0, 1)))
        self.assertTrue(bc.is_face(bc.neighbor_vector(1, 0, 0)))

    def test_iteration(self):
        bc = BoundaryCondition
        self.assertEqual(len(list(bc.foreach_neighbor_delta())), 27)

    def test_boundary_updater_v1_dirichlet(self):
        simple_solver = SimpleMultigridSolver.get_solver(["2"])
        bu = BoundaryUpdaterV1(simple_solver)
        self.assertTrue(bu.apply == bu.apply_dirichlet)

        mesh = simple_solver.fine_level.cell_values

        for i in simple_solver.fine_level.interior_points():
            mesh[i] = i[0]

        # mesh.print("before bu")

        bu.apply(simple_solver.fine_level, mesh)

        mesh.print("after bu")

        # top interior face is -4
        for index in mesh.indices():
            if index[0] == 5 and 0 < index[1] < 5 and 0 < index[2] < 5:
                self.assertEqual(mesh[index], -4.0, "mesh[{}] is {} should be -4".format(index, mesh[index]))

        # top interior face is -4
        for index in mesh.indices():
            if index[0] == 0 and 0 < index[1] < 5 and 0 < index[2] < 5:
                self.assertEqual(mesh[index], -1.0, "mesh[{}] is {} should be -1".format(index, mesh[index]))

        self.assertEqual(mesh[(5, 5, 5)], -4)  # top corner is -4
        self.assertEqual(mesh[(5, 0, 0)], -4)  # top corner is -4
        self.assertEqual(mesh[(5, 0, 5)], -4)  # top corner is -4
        self.assertEqual(mesh[(0, 0, 0)], -1)  # bottom corner is -1
        self.assertEqual(mesh[(0, 0, 5)], -1)  # bottom corner is -1
        self.assertEqual(mesh[(0, 5, 0)], -1)  # bottom corner is -1

        self.assertEqual(mesh[(5, 5, 1)], 4)  # edges are positive
        self.assertEqual(mesh[(5, 5, 2)], 4)  # edges are positive
        self.assertEqual(mesh[(5, 5, 3)], 4)  # edges are positive
        self.assertEqual(mesh[(5, 5, 4)], 4)  # edges are positive

        self.assertEqual(mesh[(0, 0, 1)], 1)  # edges are positive
        self.assertEqual(mesh[(0, 0, 2)], 1)  # edges are positive
        self.assertEqual(mesh[(0, 0, 3)], 1)  # edges are positive
        self.assertEqual(mesh[(0, 0, 4)], 1)  # edges are positive

    def test_boundary_updater_v1_periodic(self):
        simple_solver = SimpleMultigridSolver.get_solver(["2", "-bc", "p"])
        bu = BoundaryUpdaterV1(simple_solver)
        self.assertTrue(bu.apply == bu.apply_periodic)

        mesh = simple_solver.fine_level.cell_values

        for i in simple_solver.fine_level.interior_points():
            mesh[i] = i[0]

        # mesh.print("before bu")

        bu.apply(simple_solver.fine_level, mesh)

        mesh.print("after bu")

        # entire top boundary is 1, a projection of the bottom interior plane
        for index in mesh.indices():
            if index[0] == 5:
                self.assertEqual(mesh[index], 1.0, "mesh[{}] is {} should be 1".format(index, mesh[index]))

        # entire bottom boundary is 4, a projection of the top interior plane
        for index in mesh.indices():
            if index[0] == 0:
                self.assertEqual(mesh[index], 4.0, "mesh[{}] is {} should be 4".format(index, mesh[index]))

