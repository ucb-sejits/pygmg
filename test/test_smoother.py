from __future__ import print_function
import unittest
from hpgmg.finite_volume.operators.chebyshev_smoother import ChebyshevSmoother
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother

import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestApplyOp(unittest.TestCase):
    def test_jacobi_smoother(self):
        solver = simple_hpgmg.SimpleMultigridSolver.get_solver([
            "3",
            "--dimensions", "2",
            "--smoother", "j",
            "--smoother-iterations", "2",
            "--boundary-condition", "p",
        ])

        self.assertIsInstance(solver.smoother, JacobiSmoother)
        self.assertFalse(solver.smoother.use_l1_jacobi)

        base_level = solver.fine_level
        base_level.fill_mesh(base_level.cell_values, 0.0)

        base_level.fill_mesh(base_level.right_hand_side, 0.0)
        base_level.right_hand_side[(1, 1)] = 1.0
        base_level.right_hand_side.print("right hand-side")

        solver.problem_operator.rebuild_operator(base_level, source_level=None)
        self.assertEqual(solver.problem_operator.h2inv, 1.0/(base_level.h**2))
        self.assertTrue(all(base_level.d_inverse[x] == 1.0/257 for x in base_level.interior_points() ))

        mesh = base_level.cell_values
        self.assertTrue(all(mesh[x] == 0 for x in mesh.indices()), "zeros everywhere")
        solver.smoother.smooth(base_level, base_level.cell_values, base_level.right_hand_side)


        mesh.print("cell values after smooth")
        self.assertTrue(all(m == 0.0 for m in mesh[3]), "spike should not have propagated past row 3")
        self.assertTrue(all(m == 0.0 for m in mesh[:, 3]), "spike should not have propagated past col 3")

        num_smooths = 10
        last_norm = 1.0e9
        for trial in range(num_smooths):
            solver.smoother.smooth(base_level, base_level.cell_values, base_level.right_hand_side)
            norm = base_level.norm_mesh(base_level.cell_values-base_level.temp)
            self.assertLess(norm, last_norm, "trial {} norm {} should be less than last norm {} ".format(
                trial, norm, last_norm
            ))
            last_norm = norm
        base_level.cell_values.print("cell values after {} smooths".format(num_smooths*2))

    def test_chebyshev_smoother(self):
        solver = simple_hpgmg.SimpleMultigridSolver.get_solver([
            "3",
            "--dimensions", "2",
            "--smoother", "c",
            "--smoother-iterations", "2",
            "--boundary-condition", "p",
        ])

        self.assertIsInstance(solver.smoother, ChebyshevSmoother)

        base_level = solver.fine_level
        base_level.fill_mesh(base_level.cell_values, 0.0)

        base_level.fill_mesh(base_level.right_hand_side, 0.0)
        base_level.right_hand_side[(1, 1)] = 1.0
        base_level.right_hand_side.print("right hand-side")

        solver.problem_operator.rebuild_operator(base_level, source_level=None)
        self.assertEqual(solver.problem_operator.h2inv, 1.0/(base_level.h**2))
        self.assertTrue(all(base_level.d_inverse[x] == 1.0/257 for x in base_level.interior_points() ))

        mesh = base_level.cell_values
        self.assertTrue(all(mesh[x] == 0 for x in mesh.indices()), "zeros everywhere")
        solver.smoother.smooth(base_level, base_level.cell_values, base_level.right_hand_side)


        mesh.print("cell values after smooth")
        self.assertTrue(all(m == 0.0 for m in mesh[3]), "spike should not have propagated past row 3")
        self.assertTrue(all(m == 0.0 for m in mesh[:, 3]), "spike should not have propagated past col 3")

        num_smooths = 10
        last_norm = 1.0e9
        for trial in range(num_smooths):
            solver.smoother.smooth(base_level, base_level.cell_values, base_level.right_hand_side)
            norm = base_level.norm_mesh(base_level.cell_values-base_level.temp)
            self.assertLess(norm, last_norm, "trial {} norm {} should be less than last norm {} ".format(
                trial, norm, last_norm
            ))
            last_norm = norm
        base_level.cell_values.print("cell values after {} smooths".format(num_smooths*2))
