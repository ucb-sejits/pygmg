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

        variable_beta = VariableBeta(dimensions=num_dimensions)

        mesh = Mesh([size for _ in range(num_dimensions)])

        for coord in mesh.indices():
            index = coord.to_vector()

            a_beta_d, a_beta = variable_beta.evaluate_beta(index)
            b_beta_d, b_beta = variable_beta.evaluate_beta_3d(index)

            self.assertAlmostEqual(
                a_beta_d, b_beta_d,
                "at {} a_beta_d did not match b_beta_d {} != {}".format(index, a_beta_d, b_beta_d)
            )
            self.assertTrue(
                a_beta.near(b_beta),
                "at {} a_beta_d did not match b_beta_d {} != {}".format(index, a_beta, b_beta)
            )

            for d in range(num_dimensions):
                index_at_face_d = Vector(
                    v if d != dim else v - 0.5 * h for dim, v in enumerate(index)
                )

                b_beta_d, b_beta = variable_beta.evaluate_beta_3d(index_at_face_d)
                a_beta_d, a_beta = variable_beta.evaluate_beta(index_at_face_d)

                self.assertAlmostEqual(
                    a_beta_d, b_beta_d,
                    "at {} a_beta_d did not match b_beta_d {} != {}".format(index_at_face_d, a_beta_d, b_beta_d)
                )
                self.assertTrue(
                    a_beta.near(b_beta),
                    "at {} a_beta_d did not match b_beta_d {} != {}".format(index_at_face_d, a_beta, b_beta)
                )

    def test_runs_at_2d(self):
        size = 8
        num_dimensions = 2
        h = 1.0 / size

        variable_beta = VariableBeta(dimensions=num_dimensions)
        mesh = Mesh([size for _ in range(num_dimensions)])
        beta_mesh = Mesh(mesh.space)
        for coord in mesh.indices():
            index = coord.to_vector()

            beta_i, beta = variable_beta.evaluate_beta(index)
            beta_mesh[coord] = beta[1]

        for i in range(size):
            for j in range(size):
                print("{:12.4e}".format(beta_mesh[(i, j)]), end="")
            print()

