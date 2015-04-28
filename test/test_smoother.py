from __future__ import print_function
import unittest
from hpgmg.finite_volume.operators.chebyshev_smoother import ChebyshevSmoother
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother

import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestApplyOp(unittest.TestCase):
    def test_jacobi_smoother(self):
        solver = simple_hpgmg.SimpleMultigridSolver.get_solver("3 -d 2 -sm j -si 4000 -bc p".split())

        self.assertIsInstance(solver.smoother, JacobiSmoother)
        self.assertFalse(solver.smoother.use_l1_jacobi)

        base_level = solver.fine_level
        base_level.fill_mesh(base_level.cell_values, 0.0)

        # mesh = base_level.right_hand_side
        # for point in mesh.indices():
        #     mesh[point] = sum(list(point))
        base_level.fill_mesh(base_level.right_hand_side, 0.0)
        base_level.right_hand_side[(1, 1)] = 1.0
        base_level.right_hand_side.print("right hand-side")

        solver.problem_operator.rebuild_operator(base_level, source_level=None)
        self.assertEqual(solver.problem_operator.h2inv, 1.0/(base_level.h**2))
        base_level.d_inverse.print("d1 inverse")

        solver.boundary_updater.apply(base_level, base_level.cell_values)
        base_level.cell_values.print("before jacobi smoothed mesh")
        solver.smoother.smooth(base_level, base_level.cell_values, base_level.right_hand_side)
        base_level.cell_values.print("after jacobi smoothed mesh")

        # solver.boundary_updater.apply(base_level, base_level.cell_values)
        # solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
        # base_level.cell_values.print("jacobi smoothed mesh")
        #
        # solver.boundary_updater.apply(base_level, base_level.cell_values)
        # solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
        # base_level.cell_values.print("jacobi smoothed mesh")

    # def test_chebyshev_smoother(self):
    #     solver = simple_hpgmg.SimpleMultigridSolver.get_solver("1 -d 2 -sm c".split())
    #
    #     self.assertIsInstance(solver.smoother, ChebyshevSmoother)
    #     base_level = solver.fine_level
    #     mesh = base_level.cell_values
    #     for point in mesh.indices():
    #         mesh[point] = sum(list(point))
    #     mesh.print("mesh")
    #
    #     solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
    #     base_level.cell_values.print("chebyshev smoothed mesh")
"""
    def test2dApplyOp(self):
        mesh = Mesh([3, 3])
        mesh.fill(1)
        print(mesh)
        op = StencilVonNeumannR1(solver=self)
        J = JacobiSmoother(op, True, 1)
"""
