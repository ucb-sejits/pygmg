from __future__ import print_function
import unittest

from hpgmg.finite_volume.problems.problem_p6 import ProblemP6

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestProblemP6(unittest.TestCase):
    def test_generators_for_other_dimensions(self):
        for d in range(1, 5):
            problem = ProblemP6(d)
            print("Dimensions {}".format(d))
            print("    {}".format(problem.expression))
            # following is crude test that we have the right number of terms
            self.assertEqual(d-1, problem.expression.__str__().count(')*('))
