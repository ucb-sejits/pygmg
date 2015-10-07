from __future__ import print_function
import unittest

from hpgmg.finite_volume.problems.problem_p4 import ProblemP4

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestProblemP4(unittest.TestCase):
    def test_generators_for_other_dimensions(self):
        for d in range(1, 5):
            problem = ProblemP4(d)
            print("Dimensions {}".format(d))
            print("    {}".format(problem.expression))
            # following is crude test that we have the right number of terms
            self.assertEqual(d-1, problem.expression.__str__().count(')*('))

