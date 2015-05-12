from __future__ import print_function
import unittest
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.solvers.bicgstab import BiCGStab

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestBottomSolverBiCGStab(unittest.TestCase):
    def test_default_bottom_solver_is_bicgstab(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-bc", "p"])

        self.assertIsInstance(solver.bottom_solver, BiCGStab)
        bottom_solver = solver.bottom_solver
        self.assertTrue(bottom_solver.krylov_diagonal_precondition)
        self.assertTrue(bottom_solver.solver == solver)

    def test_bicgstab(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-bc", "p"])

        self.assertIsInstance(solver.bottom_solver, BiCGStab)