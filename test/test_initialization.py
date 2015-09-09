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
        solver = SimpleMultigridSolver.get_solver(["2", "-d", "2", "-vc"])

        save_beta_face_values = [
            solver.fine_level.beta_face_values[i].copy()
            for i in range(solver.dimensions)
        ]

        for i in range(solver.dimensions):
            solver.fine_level.beta_face_values[i].fill(0.0)

        solver.problem.initialize_problem_codegen(solver, solver.fine_level)

        for i in range(solver.dimensions):
            save_beta_face_values[i].print("original beta_faces_values[{}]".format(i))
            solver.fine_level.beta_face_values[i].print("codegen beta_faces_values[{}]".format(i))
            for index in solver.fine_level.indices():
                self.assertAlmostEqual(save_beta_face_values[i][index], solver.fine_level.beta_face_values[i][index],
                                       msg="face {} not same".format(i))