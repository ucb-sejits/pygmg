from __future__ import print_function
import unittest
import numpy as np
from hpgmg.finite_volume.operators.chebyshev_smoother import ChebyshevSmoother

from hpgmg.finite_volume.operators.stencil_von_neumann_r1 import StencilVonNeumannR1
from hpgmg.finite_volume.iterative_solver import IterativeSolver
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord, Vector


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestSimpleMultigridSolver(unittest.TestCase):
    def test_command_line_processing(self):
        solver = SimpleMultigridSolver.get_solver(["3"])
        self.assertEqual(solver.global_size, Coord(8, 8, 8), "default size is 2^log_2_dim_size (3 above)")
        self.assertEqual(solver.dimensions, 3, "default is 3d")

        self.assertIsInstance(solver.problem_operator, StencilVonNeumannR1,
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

    @staticmethod
    def original_initialize_3d(solver, level):
        """
        This function is deprecated, it is hard coded for 3d and is kept here at present for
        testing purposes
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0
        beta_xyz = Vector(0.0, 0.0, 0.0)
        beta_i, beta_j, beta_k = 1.0, 1.0, 1.0

        problem = solver.problem
        if level.is_variable_coefficient:
            beta_generator = solver.beta_generator

        for element_index in level.indices():
            absolute_position = level.coord_to_cell_center_point(element_index)

            if level.is_variable_coefficient:
                beta_i, _ = beta_generator.evaluate_beta(level.coord_to_face_center_point(element_index, 0))
                beta_j, _ = beta_generator.evaluate_beta(level.coord_to_face_center_point(element_index, 1))
                beta_k, _ = beta_generator.evaluate_beta(level.coord_to_face_center_point(element_index, 2))
                beta, beta_xyz = beta_generator.evaluate_beta(absolute_position)

            u, u_xyz, u_xxyyzz = problem.evaluate_u(absolute_position)

            # double F = a*A*U - b*( (Bx*Ux + By*Uy + Bz*Uz)  +  B*(Uxx + Uyy + Uzz) );
            f = solver.a * alpha * u - (
                solver.b * (
                    (beta_xyz.i * u_xyz.i + beta_xyz.j * u_xyz.j + beta_xyz.k * u_xyz.k) +
                    beta * (u_xxyyzz.i + u_xxyyzz.j + u_xxyyzz.k)
                )
            )

            level.right_hand_side[element_index] = f
            level.exact_solution[element_index] = u

            level.alpha[element_index] = alpha
            level.beta_face_values[0][element_index] = beta_i
            level.beta_face_values[1][element_index] = beta_j
            level.beta_face_values[2][element_index] = beta_k

        if (solver.a == 0.0 or solver.fine_level.alpha_is_zero) and solver.boundary_is_periodic:
            # Poisson w/ periodic BC's...
            # nominally, u shifted by any constant is still a valid solution.
            # However, by convention, we assume u sums to zero.
            mean = solver.fine_level.mean_mesh(solver.fine_level.exact_solution)
            solver.fine_level.shift_mesh(solver.fine_level.exact_solution, -mean, solver.fine_level.exact_solution)

        if solver.boundary_is_periodic:
            average_value_of_rhs = solver.fine_level.mean_mesh(solver.fine_level.right_hand_side)
            if average_value_of_rhs != 0.0:
                solver.fine_level.shift_mesh(
                    solver.fine_level.right_hand_side,
                    average_value_of_rhs,
                    solver.fine_level.right_hand_side
                )

        solver.problem_operator.rebuild_operator(solver.fine_level, source_level=None)

    def test_backward_compatibility_of_initializer(self):
        solver = SimpleMultigridSolver.get_solver(["2"])

        save_right_hand_side = np.copy(solver.fine_level.right_hand_side)
        save_exact_solution = np.copy(solver.fine_level.exact_solution)

        save_beta_arrays = [np.copy(solver.fine_level.beta_face_values[x]) for x in range(solver.dimensions)]
        for x in range(solver.dimensions):
            solver.fine_level.fill_mesh(solver.fine_level.beta_face_values[x], 0.0)

        TestSimpleMultigridSolver.original_initialize_3d(solver, solver.fine_level)

        for index in solver.fine_level.indices():
            x = save_exact_solution[index]
            y = solver.fine_level.exact_solution[index]

            self.assertAlmostEqual(x, y, msg="at {} current {} original {}".format(str(index), x, y))
            for face_index in range(solver.dimensions):
                self.assertEqual(
                    save_beta_arrays[face_index][index],
                    solver.fine_level.beta_face_values[face_index][index]
                )

    def test_coordinate_to_vector_stuff(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2"])

        level = solver.fine_level

        print("cell center points for 2^3 mesh with ghost zone, in 2d")
        for i in range(level.space[0]):
            print("{:4d}  ".format(i), end="")
            for j in range(level.space[1]):
                point = level.coord_to_cell_center_point(Coord(i, j))
                print("({:7.4f}, {:7.4f})".format(point[0], point[1]), end=" ")
            print()
        print()

        f, c, v = level.coord_to_cell_center_point, Coord, Vector
        self.assertEqual(f(c(0, 0)), v(-0.0625, -0.0625))
        self.assertEqual(f(c(5, 3)), v(0.5625, 0.3125))

        print("dim 0 face points for 2^3 mesh with ghost zone, in 2d")
        for i in range(level.space[0]):
            print("{:4d}  ".format(i), end="")
            for j in range(level.space[1]):
                point = level.coord_to_face_center_point(Coord(i, j), 0)
                print("({:7.4f}, {:7.4f})".format(point[0], point[1]), end=" ")
            print()
        print()

        f, c, v = level.coord_to_face_center_point, Coord, Vector
        self.assertEqual(f(c(1, 1), 0), v(0.0000, 0.0625))
        self.assertEqual(f(c(8, 8), 0), v(0.8750, 0.9375))

    def test_operator_construction(self):
        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2"])
        self.assertIsInstance(solver.smoother, JacobiSmoother)

        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-sm", "j"])
        self.assertIsInstance(solver.smoother, JacobiSmoother)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_constant_coefficient_unfused_boundary_conditions
        )

        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-sm", "c"])
        self.assertIsInstance(solver.smoother, ChebyshevSmoother)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_constant_coefficient_unfused_boundary_conditions
        )

        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-sm", "c"])
        self.assertIsInstance(solver.smoother, ChebyshevSmoother)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_constant_coefficient_unfused_boundary_conditions
        )

        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-sm", "c", "-vc"])
        self.assertIsInstance(solver.smoother, ChebyshevSmoother)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_variable_coefficient_unfused_boundary_conditions_helmholtz
        )

        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-sm", "c", "-vc", "-eq", "h"])
        self.assertIsInstance(solver.smoother, ChebyshevSmoother)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_variable_coefficient_unfused_boundary_conditions_helmholtz
        )

        solver = SimpleMultigridSolver.get_solver(["3", "-d", "2", "-sm", "c", "-vc", "-eq", "p"])
        self.assertIsInstance(solver.smoother, ChebyshevSmoother)
        self.assertEqual(
            solver.problem_operator.apply_op,
            solver.problem_operator.apply_op_variable_coefficient_unfused_boundary_conditions_poisson
        )
