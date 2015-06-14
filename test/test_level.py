from __future__ import print_function
import unittest
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestSimpleLevel(unittest.TestCase):
    def setUp(self):
        self.solver = SimpleMultigridSolver.get_solver(["2", "-d", "2"])

    def test_basics(self):
        solver = self.solver
        level = solver.fine_level

        self.assertEqual(solver, level.solver)
        self.assertEqual(solver.global_size, level.space - level.ghost_zone*2)
        self.assertEqual(level.level_number, 0)

    def test_add_meshes(self):
        """
        add meshes only works on interior points
        :return:
        """
        level = self.solver.fine_level
        mesh1, mesh2 = level.cell_values, level.exact_solution

        for index in level.interior_points():
            mesh1[index] = 1.0
            mesh2[index] = 4.0

        level.add_meshes(level.temp, 1.0, mesh1, 1.0, mesh2)
        self.assertTrue(all(level.temp[index] == 5.0 for index in level.interior_points()))
        level.add_meshes(level.temp, 2.0, mesh1, 0.5, mesh2)
        self.assertTrue(all(level.temp[index] == 4.0 for index in level.interior_points()))
        level.add_meshes(level.temp, 2.0, mesh1, -1.0, mesh2)
        self.assertTrue(all(level.temp[index] == -2.0 for index in level.interior_points()))

    def test_multiply_meshes(self):
        """
        element-wise multiply of meshes with scaling factor
        :return:
        """
        level = self.solver.fine_level
        mesh1, mesh2 = level.cell_values, level.exact_solution

        for index in level.interior_points():
            mesh1[index] = 0.04
            mesh2[index] = 0.2

        level.multiply_meshes(level.temp, 1.0, mesh1, mesh2)
        self.assertTrue(all(level.temp[index] == 0.008 for index in level.interior_points()))
        level.multiply_meshes(level.temp, 1000.0, mesh1, mesh2)
        self.assertTrue(all(level.temp[index] == 8.0 for index in level.interior_points()))

    def test_mean_meshes(self):
        level = self.solver.fine_level
        mesh1, mesh2 = level.cell_values, level.exact_solution

        for index in level.interior_points():
            mesh1[index] = sum(index)
            mesh2[index] = 11.0

        mean = level.mean_mesh(mesh1)
        self.assertEqual(mean, 5.0)

        self.assertEqual(level.mean_mesh(mesh2), 11.0)

    def test_invert_mesh(self):
        level = self.solver.fine_level
        mesh1, mesh2 = level.cell_values, level.exact_solution

        for index in level.interior_points():
            value = float(sum(index))
            mesh1[index] = value
            mesh2[index] = 1.0 / value

        level.invert_mesh(level.temp, 1.0, mesh1)
        self.assertTrue(level.meshes_interiors_equal(level.temp, mesh2))

    def test_dot_mesh(self):
        level = self.solver.fine_level
        mesh1, mesh2 = level.cell_values, level.exact_solution

        for index in level.interior_points():
            value = float(sum(index))
            mesh1[index] = 1.0
            mesh2[index] = 2.0

        total = level.dot_mesh(mesh1, mesh2)
        self.assertTrue(total, 12.0)

    def test_norm_mesh(self):
        level = self.solver.fine_level
        mesh1, mesh2 = level.cell_values, level.exact_solution

        for index in level.interior_points():
            value = float(sum(index))
            mesh1[index] = float(sum(index))
            mesh2[index] = -float(sum(index)) * 2.0

        total = level.norm_mesh(mesh1)
        self.assertTrue(total, 4.0)
        total = level.norm_mesh(mesh2)
        self.assertTrue(total, 8.0)








