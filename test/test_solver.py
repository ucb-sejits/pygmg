from __future__ import print_function
import unittest
from hpgmg.finite_volume.Simple7PointOperator import SimpleConstantCoefficientOperator
from hpgmg.finite_volume.iterative_solver import IterativeSolver
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestSimpleMultigridSolver(unittest.TestCase):
    def test_command_line_processing(self):
        solver = SimpleMultigridSolver.get_solver("3")
        self.assertEqual(solver.global_size, Coord(8, 8, 8), "default size is 2^log_2_dim_size (3 above)")
        self.assertEqual(solver.dimensions, 3, "default is 3d")

        self.assertIsInstance(solver.problem_operator, SimpleConstantCoefficientOperator,
                              "default is simple 7pt stencil")
        self.assertIsInstance(solver.smoother, JacobiSmoother, "default smoother is Jacobi")
        self.assertIsInstance(solver.bottom_solver, IterativeSolver, "default bottom_solver")
        self.assertFalse(solver.boundary_is_periodic, "default boundary is not periodic")
        self.assertTrue(solver.boundary_is_dirichlet, "default boundary is dirichlet")
        self.assertFalse(solver.is_variable_coefficient, "default is fixed beta")

        self.assertEqual(solver.a, 1.0, "default is helmholtz with a == 1.0")
        self.assertEqual(solver.b, 1.0, "default is helmholtz with b == 1.0")
        self.assertEqual(solver.ghost_zone, Coord(1, 1, 1), "default ghost zone is 1 in all dimensions")

        self.assertIsInstance(solver.fine_level, SimpleLevel, "solver initialized with a default level")
