from __future__ import print_function
import unittest

import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestApplyOp(unittest.TestCase):
    def test_smoother(self):
        solver = simple_hpgmg.SimpleMultigridSolver.get_solver("1".split())
        base_level = solver.fine_level
        mesh = base_level.cell_values
        for point in mesh.indices():
            mesh[point] = sum(list(point))
        mesh.print("mesh")

        solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
        base_level.cell_values.print("smoothed mesh")
"""
    def test2dApplyOp(self):
        mesh = Mesh([3, 3])
        mesh.fill(1)
        print(mesh)
        op = StencilVonNeumannR1(solver=self)
        J = JacobiSmoother(op, True, 1)
"""
