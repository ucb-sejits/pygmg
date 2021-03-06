from __future__ import print_function
import unittest
from math import sin, cos
import numpy as np

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.problems.problem_sine_n_dim import SineProblemND
from hpgmg.finite_volume.space import Vector, Space

import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestProblemSineND(unittest.TestCase):
    # TODO: figure out if this is a computational problem
    @unittest.skip('This needs to be verified as not a computational problem')
    def test_problem_sine_n_d_matches_hard_coded_sine_problem(self):
        number_of_dimensions = 3
        solver = simple_hpgmg.SimpleMultigridSolver.get_solver([
            "3",
            "--dimensions", "3",
            "--smoother", "j",
            "--problem", "sine"
        ])
        problem = SineProblemND(number_of_dimensions)
        print("function python source")
        print("    {}".format(problem.expression))
        space = Space(10 for _ in range(number_of_dimensions))
        mesh = Mesh(space)

        level = solver.fine_level

        #fill U
        solver.initialize_mesh(level, level.exact_solution, problem.expression, level.coord_to_cell_center_point)

        if number_of_dimensions == 3:
            for index in solver.fine_level.interior_points():
                point = Vector(float(index[d]) / mesh.space[d] for d in range(mesh.space.ndim))

                # a, da, d2a = problem.evaluate_u(point)
                b, db, d2b = SineProblem.evaluate_u(point)

                self.assertAlmostEqual(level.exact_solution[index], b,
                                       msg="index {}, new {:10.4f} old {:10.4f}".format(
                                           index.__repr__(), level.exact_solution[index], b
                ))
                # self.assertTrue(da.near(db, threshold=1e-6), "mismatch du {:12} {:12}".format(da, db))
                # self.assertTrue(d2a.near(d2b, threshold=1e-6), "mismatch d2u {:12} {:12}".format(d2a, d2b))

    def test_generators_for_other_dimensions(self):
        for d in range(1, 5):
            problem = SineProblemND(d)
            print("Dimensions {}".format(d))
            print("    {}".format(problem.expression))
            # following is crude test that we have the right number of terms
            self.assertEqual(d*2, problem.expression.__str__().count('**13'))


class SineProblem(object):
    """
    This was the original fixed 3d implementation of the sine problem
    """
    alpha = 1.0
    beta = 1.0

    @staticmethod
    def evaluate_u(coord):
        """
        compute the exact value of the function u for a given coordinate
        :param coord:
        :return: value of u and a tuple of u for each dimension
        """
        x, y, z = coord

        c1 = 2.0*np.pi
        c2 = 6.0*np.pi
        p = 13  # must be odd(?) and allows up to p-2 order MG
        u = (sin(c1*x)**p) * (sin(c1*y)**p) * (sin(c1*z)**p)
        u_x = c1*p*cos(c1*x) * (sin(c1*x)**(p-1)) * (sin(c1*y)**p) * (sin(c1*z)**p)
        u_y = c1*p*cos(c1*y) * (sin(c1*y)**(p-1)) * (sin(c1*x)**p) * (sin(c1*z)**p)
        u_z = c1*p*cos(c1*z) * (sin(c1*z)**(p-1)) * (sin(c1*x)**p) * (sin(c1*y)**p)

        u_xx = c1*c1*p * ((p-1) * (sin(c1*x)**(p-2)) * (cos(c1*x)**2) - sin(c1*x)**p) * (sin(c1*y)**p) * (sin(c1*z)**p)
        u_yy = c1*c1*p * ((p-1) * (sin(c1*y)**(p-2)) * (cos(c1*y)**2) - sin(c1*y)**p) * (sin(c1*x)**p) * (sin(c1*z)**p)
        u_zz = c1*c1*p * ((p-1) * (sin(c1*z)**(p-2)) * (cos(c1*z)**2) - sin(c1*z)**p) * (sin(c1*x)**p) * (sin(c1*y)**p)

        u += (sin(c2*x)**p) * (sin(c2*y)**p) * (sin(c2*z)**p)
        u_x += c2*p*cos(c2*x) * (sin(c2*x)**(p-1)) * (sin(c2*y)**p) * (sin(c2*z)**p)
        u_y += c2*p*cos(c2*y) * (sin(c2*y)**(p-1)) * (sin(c2*x)**p) * (sin(c2*z)**p)
        u_z += c2*p*cos(c2*z) * (sin(c2*z)**(p-1)) * (sin(c2*x)**p) * (sin(c2*y)**p)
        u_xx += c2*c2*p * ((p-1)*(sin(c2*x)**(p-2)) * (cos(c2*x)**2) - sin(c2*x)**p) * (sin(c2*y)**p) * (sin(c2*z)**p)
        u_yy += c2*c2*p * ((p-1)*(sin(c2*y)**(p-2)) * (cos(c2*y)**2) - sin(c2*y)**p) * (sin(c2*x)**p) * (sin(c2*z)**p)
        u_zz += c2*c2*p * ((p-1)*(sin(c2*z)**(p-2)) * (cos(c2*z)**2) - sin(c2*z)**p) * (sin(c2*x)**p) * (sin(c2*y)**p)

        return u, Vector(u_x, u_y, u_z), Vector(u_xx, u_yy, u_zz)
