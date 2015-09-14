from __future__ import print_function
__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from math import pow, pi
from sympy import symbols, sin, cos, Symbol, lambdify

from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver


class TestInitialization(unittest.TestCase):
    def test_initialization_problem_init_codegen(self):
        solver = SimpleMultigridSolver.get_solver(["2", "-d", "3", "-vc"])

        save_beta_face_values = [
            solver.fine_level.beta_face_values[i].copy()
            for i in range(solver.dimensions)
        ]

        for i in range(solver.dimensions):
            solver.fine_level.beta_face_values[i].fill(0.0)

        solver.problem.initialize_problem_codegen(solver, solver.fine_level)

        for i in range(solver.dimensions):
            # save_beta_face_values[i].print("original beta_faces_values[{}]".format(i))
            # solver.fine_level.beta_face_values[i].print("codegen beta_faces_values[{}]".format(i))
            for index in solver.fine_level.indices():
                self.assertAlmostEqual(save_beta_face_values[i][index], solver.fine_level.beta_face_values[i][index],
                                       msg="face {} not same".format(i))

    def test_initialization_problem_init_codegen_fv(self):
        solver = SimpleMultigridSolver.get_solver(["2", "-d", "2", "-vc", "--problem", "fv"])

        save_beta_face_values = [
            solver.fine_level.beta_face_values[i].copy()
            for i in range(solver.dimensions)
        ]

        for i in range(solver.dimensions):
            solver.fine_level.beta_face_values[i].fill(0.0)

        solver.problem.initialize_problem_codegen(solver, solver.fine_level)

        for i in range(solver.dimensions):
            # save_beta_face_values[i].print("original beta_faces_values[{}]".format(i))
            # solver.fine_level.beta_face_values[i].print("codegen beta_faces_values[{}]".format(i))
            for index in solver.fine_level.indices():
                self.assertAlmostEqual(save_beta_face_values[i][index], solver.fine_level.beta_face_values[i][index],
                                       msg="face {} not same".format(i))

    def test_compare_sympy_hpgmg_beta_corrections(self):
        period, power = 2 * pi, 7.0
        s_period, s_power = symbols("period power")

        s_x, s_y, s_z = symbols("x y z")

        a, p = s_period, s_power
        sympy_expression = sin(a*s_x)**p * sin(a*s_y)**p * sin(a*s_z)**p
        print("Sympy expression => {}".format(sympy_expression))

        sym_f = lambdify((s_x, s_y, s_z, a, p), sympy_expression)

        f_xx_expression = sympy_expression.diff(Symbol("x"), 2)
        print("f_xx {}".format(f_xx_expression))

        sym_f_xx = lambdify((s_x, s_y, s_z, a, p), f_xx_expression)

        print("symp_f(0.1, 0.2, 0.3) => {}".format(sym_f(0.1, 0.2, 0.3, period, power)))
        print("type symp_f(0.1, 0.2, 0.3) => {}".format(type(sym_f(0.1, 0.2, 0.3, period, power))))
        print("symp_f_xx(0.1, 0.2, 0.3) => {}".format(sym_f_xx(0.1, 0.2, 0.3, period, power)))

        a, p = period, power

        def sam_f(x, y, z):
            # noinspection PyPep8
            return pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  # directly from HPGMG

        def sam_f_xx(x, y, z):
            # noinspection PyPep8
            return -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p-2)*pow(sin(a*y),p  )*pow(sin(a*z),p  )*pow(cos(a*x),2)  # Directly from HPGMG

        print("sam_f(0.1, 0.2, 0.3) => {}".format(sam_f(0.1, 0.2, 0.3)))
        print("type sam_f(0.1, 0.2, 0.3) => {}".format(type(sam_f(0.1, 0.2, 0.3))))
        print("sam_f_xx(0.1, 0.2, 0.3) => {}".format(sam_f_xx(0.1, 0.2, 0.3)))

        solver = SimpleMultigridSolver.get_solver(["2", "-d", "3"])

        for index in solver.fine_level.interior_points():
            i_x, i_y, i_z = solver.fine_level.coord_to_cell_center_point(index)

            sym_val = sym_f(i_x, i_y, i_z, period, power)
            sam_val = sam_f(i_x, i_y, i_z)
            self.assertAlmostEqual(sym_val, sam_val)

            self.assertAlmostEqual(sym_f_xx(i_x, i_y, i_z, period, power), sam_f_xx(i_x, i_y, i_z),
                                   msg="at {} :: {} <> {}".format(
                                       (i_x, i_y, i_z),
                                       sym_f_xx(i_x, i_y, i_z, period, power), sam_f_xx(i_x, i_y, i_z),
                                   ))