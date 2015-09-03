from __future__ import print_function
import unittest
import numpy as np
from math import sin
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.operators.variable_beta_generators import VariableBeta

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestVariableBetaGenerators(unittest.TestCase):
    def test_n_d_beta_generator_matches_3d(self):
        solver = SimpleMultigridSolver.get_solver(["4", "-d", "3"])
        level = solver.fine_level

        variable_beta = VariableBeta(dimensions=solver.dimensions)

        for coord in level.indices():
            point = level.coord_to_cell_center_point(coord)

            a_beta_d = variable_beta.evaluate_beta(point)
            b_beta_d, b_beta = variable_beta.evaluate_beta_3d(point)

            self.assertAlmostEqual(
                a_beta_d, b_beta_d,
                "at {} a_beta_d did not match b_beta_d {} != {}".format(point, a_beta_d, b_beta_d)
            )
            # self.assertTrue(
            #     a_beta.near(b_beta),
            #     "at {} a_beta_d did not match b_beta_d {} != {}".format(point, a_beta, b_beta)
            # )

            for d in range(solver.dimensions):
                point_at_face_d = level.coord_to_face_center_point(coord, d)

                b_beta_d, b_beta = variable_beta.evaluate_beta_3d(point_at_face_d)
                a_beta_d = variable_beta.evaluate_beta(point_at_face_d)

                self.assertAlmostEqual(
                    a_beta_d, b_beta_d,
                    "at {} a_beta_d did not match b_beta_d {} != {}".format(point_at_face_d, a_beta_d, b_beta_d)
                )
                # self.assertTrue(
                #     a_beta.near(b_beta),
                #     "at {} a_beta_d did not match b_beta_d {} != {}".format(point_at_face_d, a_beta, b_beta)
                # )

    def test_runs_at_2d(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-vc"])
        level = solver.fine_level

        variable_beta = VariableBeta(dimensions=solver.dimensions)

        for dim in range(solver.dimensions):
            level.beta_face_values[dim].print("beta_face_values[{}]".format(dim))

        for coord in level.indices():
            for dim in range(solver.dimensions):
                face_point = level.coord_to_face_center_point(coord, dim)
                beta_face = variable_beta.evaluate_beta(face_point)

                self.assertEqual(
                    beta_face, level.beta_face_values[dim][coord],
                    "beta face point wrong index {} face_point {}, calc {}, in level {}".format(
                        coord, face_point, beta_face, level.beta_face_values[dim][coord]
                    ))

    def williams_beta_with_corrections(self, x, y, z, h):
        # this is near exact translation from hpgmg.c
        b = 0.25
        a = 2.0 * np.pi
        first_term = 1.0 + b*sin(a*x)*sin(a*y)*sin(a*z)
        corrections = 0.0
        corrections += -a*a*b*sin(a*x)*sin(a*y)*sin(a*z) * (h*h)/24.0
        corrections += -a*a*b*sin(a*x)*sin(a*y)*sin(a*z) * (h*h)/24.0
        corrections += -a*a*b*sin(a*x)*sin(a*y)*sin(a*z) * (h*h)/24.0
        return first_term + corrections

    def test_variable_beta_for_fv_problems(self):
        solver = SimpleMultigridSolver.get_solver(["2", "-d", "3", "-vc", "--problem", "fv"])
        level = solver.fine_level

        variable_beta = VariableBeta(dimensions=solver.dimensions)
        beta_expression = variable_beta.get_beta_fv_expression(add_4th_order_correction=False)

        williams_formula_3d = "0.25*sin(6.28318530718*x0)*sin(6.28318530718*x1)*sin(6.28318530718*x2) + 1.0"

        self.assertEqual(beta_expression.__repr__(), williams_formula_3d)

        beta_expression = variable_beta.get_beta_fv_expression(add_4th_order_correction=True, cell_size=level.cell_size)

        func = solver.problem.get_func(beta_expression, ("x0", "x1", "x2"))

        for ix in range(10):
            for iy in range(10):
                for iz in range(10):
                    x, y, z = ix/10.0, iy/10.0, iz/10.0
                    williams_beta_value = self.williams_beta_with_corrections(x, y, z, level.cell_size)
                    pygmg_value = func(x, y, z)

                    self.assertAlmostEqual(williams_beta_value, pygmg_value)
