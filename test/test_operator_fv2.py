from __future__ import print_function
import unittest
from hpgmg.finite_volume.operators.operators_fv2 import OperatorsFV2
from hpgmg.finite_volume.space import Coord
from hpgmg.finite_volume.operators.stencil_von_neumann_r1 import StencilVonNeumannR1
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestOperatorFV2(unittest.TestCase):

    def test_initialize_solver_with_fv2(self):
        solver = SimpleMultigridSolver.get_solver(["3", "--dimensions", "2"])

        in_mesh = solver.fine_level.cell_values
        solver.fine_level.fill_mesh(in_mesh, 1.0)

        self.assertIsInstance(solver.problem_operator, OperatorsFV2)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_constant_coefficient
        )

        self.assertEqual(solver.a, 1.0)
        self.assertEqual(solver.b, 1.0)

        solver.problem_operator.set_scale(solver.fine_level.h)
        self.assertEqual(solver.problem_operator.h2inv, 64.0)

    def test_apply_op_variable_coefficient_d2(self):
        solver = SimpleMultigridSolver.get_solver(["3", "--dimensions", "2", "-vc"])

        in_mesh = solver.fine_level.cell_values
        level = solver.fine_level
        level.fill_mesh(in_mesh, 1.0)
        for dim in in_mesh.dim_range():
            level.fill_mesh(level.beta_face_values[dim], 1.0)

        problem_operator = OperatorsFV2(solver)
        self.assertEqual(
            problem_operator.apply_op,
            problem_operator.apply_op_variable_coefficient
        )

        one_one = Coord(1, 1)
        in_mesh[one_one] = 2.0
        value = problem_operator.apply_op(in_mesh, one_one, level)
        self.assertEqual(value, 6.0)

        for dim in in_mesh.dim_range():
            level.fill_mesh(level.beta_face_values[dim], 0.5)
        value = problem_operator.apply_op(in_mesh, one_one, level)
        self.assertEqual(value, 4.0)

        in_mesh[one_one + 1] = 10.0  # diagonal neighbor change has no effect on apply_op
        value = problem_operator.apply_op(in_mesh, one_one, level)
        self.assertEqual(value, 4.0)
        in_mesh[one_one + 1] = 1.0  # diagonal neighbor change has no effect on apply_op

        in_mesh[one_one + Coord(1, 0)] = 0.0  # facing neighbor change affects stencil result
        value = problem_operator.apply_op(in_mesh, one_one, level)
        self.assertEqual(value, 4.5)

        in_mesh[one_one + Coord(-1, 0)] = 2.0  # opposite facing neighbor balances out stencil
        value = problem_operator.apply_op(in_mesh, one_one, level)
        self.assertEqual(value, 4.0)

        self.assertRaises(IndexError, problem_operator.apply_op, in_mesh, in_mesh.space - 1, level)

    def test_apply_op_constant_coefficient(self):
        solver = SimpleMultigridSolver.get_solver(["3", "--dimensions", "2"])

        in_mesh = solver.fine_level.cell_values
        solver.fine_level.fill_mesh(in_mesh, 1.0)

        self.assertIsInstance(solver.problem_operator, OperatorsFV2)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_constant_coefficient
        )
        self.assertEqual(solver.a, 1.0)
        self.assertEqual(solver.b, 1.0)

        solver.problem_operator.set_scale(solver.fine_level.h)
        self.assertEqual(solver.problem_operator.h2inv, 64.0)

        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)

        in_mesh[Coord(1, 1) + 1] = 10.0  # diagonal neighbor change has no effect on apply_op
        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)
        in_mesh[Coord(1, 1) + 1] = 1.0  # diagonal neighbor change has no effect on apply_op

        in_mesh[Coord(1, 1) + Coord(1, 0)] = 0.0  # facing neighbor change affects stencil result
        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 65.0)

        in_mesh[Coord(1, 1) + Coord(-1, 0)] = 2.0  # opposite facing neighbor balances out stencil
        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)

        value = solver.problem_operator.apply_op(in_mesh, Coord(1, 1))
        self.assertEqual(value, 1.0)

        self.assertRaises(IndexError, solver.problem_operator.apply_op, in_mesh, in_mesh.space - 1)