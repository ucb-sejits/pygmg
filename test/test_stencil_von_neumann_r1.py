from __future__ import print_function
import unittest
from hpgmg.finite_volume.space import Coord
from hpgmg.finite_volume.operators.stencil_von_neumann_r1 import StencilVonNeumannR1
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestStencilVonNeumannR1(unittest.TestCase):
    def test_apply_op_constant_coefficient_unfused_boundary_conditions(self):
        solver = SimpleMultigridSolver.get_solver(["3", "--dimensions", "2"])

        in_mesh = solver.fine_level.cell_values
        solver.fine_level.fill_mesh(in_mesh, 1.0)

        self.assertIsInstance(solver.problem_operator, StencilVonNeumannR1)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_constant_coefficient_unfused_boundary_conditions
        )
        self.assertEqual(solver.a, 1.0)
        self.assertEqual(solver.b, 1.0)
        self.assertEqual(solver.problem_operator.h2inv, 64.0)  # TODO: confirm this value

        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)

        in_mesh[Coord(1, 1) + 1] = 10.0  # diagonal neighbor change has no effect on apply_op
        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)
        in_mesh[Coord(1, 1) + 1] = 1.0  # diagonal neighbor change has no effect on apply_op

        in_mesh[Coord(1, 1) + Coord(1, 0)] = 0.0  # facing neighbor change affects stencil result
        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 65.0)  # TODO: confirm this value

        in_mesh[Coord(1, 1) + Coord(-1, 0)] = 2.0  # opposite facing neighbor balances out stencil
        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)  # TODO: confirm this value

        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)

        self.assertRaises(IndexError, solver.problem_operator.apply_op, in_mesh, in_mesh.space - 1)