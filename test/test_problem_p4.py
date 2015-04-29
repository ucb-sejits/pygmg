from __future__ import print_function
import unittest

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.problems.problem_p4 import ProblemP4
from hpgmg.finite_volume.space import Vector, Space


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestProblemP4(unittest.TestCase):
    def test_generators_for_other_dimensions(self):
        for d in range(1, 5):
            problem = ProblemP4(d)
            print("Dimensions {}".format(d))
            for line in problem.source:
                print("    {}".format(line))
