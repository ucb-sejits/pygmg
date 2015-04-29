from __future__ import print_function
import unittest
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.space import Vector
from hpgmg.finite_volume.operators.variable_beta_generators import VariableBeta

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestVariableBetaGenerators(unittest.TestCase):
    def test_n_d_beta_generator_matches_3d(self):

        size = 16
        num_dimensions = 3
        h = 1.0 / size
        half_cell = Vector([0.5 for _ in range(num_dimensions)])

        variable_beta = VariableBeta(dimensions=num_dimensions)

        mesh = Mesh([size for _ in range(num_dimensions)])

        for coord in mesh.indices():
            point = (coord.to_vector() + half_cell) * h

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

            for d in range(num_dimensions):
                point_at_face_d = Vector(
                    v if d != dim else v - 0.5 * h for dim, v in enumerate(point)
                )

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
        size = 8
        num_dimensions = 2
        h = 1.0 / size
        half_cell = Vector([0.5 for _ in range(num_dimensions)])

        variable_beta = VariableBeta(dimensions=num_dimensions)
        mesh = Mesh([size for _ in range(num_dimensions)])
        beta_face_values = [Mesh(mesh.space) for _ in range(num_dimensions)]

        beta_mesh = Mesh(mesh.space)
        for coord in mesh.indices():
            point = (coord.to_vector() + half_cell) * h

            beta_i, beta = variable_beta.evaluate_beta(point)

            beta_mesh
            for face_id in range(num_dimensions):
                beta_face_values[face_id][coord] = beta[face_id]

        for face_id in range(num_dimensions):
            beta_face_values[face_id].print("beta_face[{}]".format(face_id))
