from __future__ import print_function
import unittest

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.problems.problem_sine_n_dim import SineProblemND
from hpgmg.finite_volume.problems.problem_sine import SineProblem
from hpgmg.finite_volume.space import Vector, Space


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestProblemSineND(unittest.TestCase):
    def test_problem_sine_n_d_matches_hard_coded_sine_problem(self):
        number_of_dimensions = 3
        problem = SineProblemND(number_of_dimensions)
        print("function python source")
        for line in problem.source:
            print("    {}".format(line))
        space = Space(4 for _ in range(number_of_dimensions))
        mesh = Mesh(space)

        if number_of_dimensions == 3:
            count = 0
            for index in mesh.indices():
                point = Vector(float(index[d]) / mesh.space[d] for d in range(mesh.space.ndim))

                a, da, d2a = problem.evaluate_u(point)
                b, db, d2b = SineProblem.evaluate_u(point)

                self.assertAlmostEqual(a, b)
                self.assertTrue(da.near(db, threshold=1e-6), "mismatch du {:12} {:12}".format(da, db))
                self.assertTrue(d2a.near(d2b, threshold=1e-6), "mismatch d2u {:12} {:12}".format(d2a, d2b))

    def test_generators_for_other_dimensions(self):
        for d in range(1, 5):
            problem = SineProblemND(d)
            print("Dimensions {}".format(d))
            for line in problem.source:
                print("    {}".format(line))
