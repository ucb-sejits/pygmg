from __future__ import print_function
import numpy
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.operators.interpolation import InterpolatorPQ, InterpolatorPC
from hpgmg.finite_volume.space import Space, Coord


class TestInitialization(unittest.TestCase):
    def test_initialization_coordinate_transforms(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-vc"])

        solver.problem.initialize_problem_codegen(solver, solver.fine_level)