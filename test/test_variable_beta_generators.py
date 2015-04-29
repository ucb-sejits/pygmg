from __future__ import print_function
import unittest
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.space import Vector
from hpgmg.finite_volume.operators.variable_beta_generators import VariableBeta

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestVariableBetaGenerators(unittest.TestCase):
    def test_n_d_beta_generator_matches_3d(self):
        solver = SimpleMultigridSolver.get_solver(["4", "-d", "3"])
        level = solver.fine_level

        variable_beta = VariableBeta(dimensions=solver.dimensions)

        for coord in level.indices():
            point = level.coord_to_cell_center_point(coord)

            a_beta_d, a_beta = variable_beta.evaluate_beta(point)
            b_beta_d, b_beta = variable_beta.evaluate_beta_3d(point)

            self.assertAlmostEqual(
                a_beta_d, b_beta_d,
                "at {} a_beta_d did not match b_beta_d {} != {}".format(point, a_beta_d, b_beta_d)
            )
            self.assertTrue(
                a_beta.near(b_beta),
                "at {} a_beta_d did not match b_beta_d {} != {}".format(point, a_beta, b_beta)
            )

            for d in range(solver.dimensions):
                point_at_face_d = level.coord_to_face_center_point(coord, d)

                b_beta_d, b_beta = variable_beta.evaluate_beta_3d(point_at_face_d)
                a_beta_d, a_beta = variable_beta.evaluate_beta(point_at_face_d)

                self.assertAlmostEqual(
                    a_beta_d, b_beta_d,
                    "at {} a_beta_d did not match b_beta_d {} != {}".format(point_at_face_d, a_beta_d, b_beta_d)
                )
                self.assertTrue(
                    a_beta.near(b_beta),
                    "at {} a_beta_d did not match b_beta_d {} != {}".format(point_at_face_d, a_beta, b_beta)
                )

    def test_runs_at_2d(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-vc"])
        level = solver.fine_level

        variable_beta = VariableBeta(dimensions=solver.dimensions)

        for dim in range(solver.dimensions):
            level.beta_face_values[dim].print("beta_face_values[{}]".format(dim))

        for coord in level.indices():
            point = level.coord_to_cell_center_point(coord)

            for dim in range(solver.dimensions):
                face_point = level.coord_to_face_center_point(coord, dim)
                beta_face, _ = variable_beta.evaluate_beta(face_point)

                self.assertEqual(beta_face, level.beta_face_values[dim][coord])
