"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function
import argparse
import os
from hpgmg.finite_volume.operators.interpolation import InterpolatorPC
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother, ShivJacobi
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.operators.stencil_operators import ConstantCoefficient7pt

__author__ = 'nzhang-dev'

from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.problem_sine import SineProblem
from hpgmg.finite_volume.smoothers import gauss_siedel, jacobi_stencil


class SimpleLevel(object):
    FACE_I = 0
    FACE_J = 0
    FACE_K = 0

    def __init__(self, space, level_number=0, configuration=None):
        self.space = space
        self.configuration = configuration
        self.is_variable_coefficient = not configuration.fixed_beta
        self.problem_name = configuration.problem
        self.level_number = level_number
        if self.problem_name == 'sine':
            self.problem = SineProblem

        self.cell_values = Mesh(space)
        self.alpha = Mesh(space)
        self.beta_face_values = [
            Mesh(space),
            Mesh(space),
            Mesh(space),
        ]
        if configuration.fixed_beta:
            self.operator = ConstantCoefficient7pt()
        self.d_inverse = Mesh(space)
        self.l1_inverse = Mesh(space)
        self.temp = Mesh(space)
        self.dominant_eigen_value_of_d_inv_a = 0.0

        if self.level_number == 0:
            self.true_solution = Mesh(space)

        self.cell_size = 1.0 / space[0]

    def make_coarser_level(self):
        coarser_level = SimpleLevel(self.space//2, self.level_number+1, self.configuration)
        return coarser_level

    @property
    def h(self):
        return self.cell_size

    def initialize(self, a=1.0, b=1.0):
        alpha = 1.0
        beta = 1.0
        beta_xyz = Vector(0.0, 0.0, 0.0)
        beta_i, beta_j, beta_k = 1.0, 1.0, 1.0

        problem = self.problem

        for element_index in self.space.points:
            half_cell = Vector([0.5 for _ in self.space])
            absolute_position = (Vector(element_index) + half_cell) * self.cell_size

            if self.is_variable_coefficient:
                beta_i, _ = problem.evaluate_beta(absolute_position-Vector(self.h*0.5, 0.0, 0.0))
                beta_j, _ = problem.evaluate_beta(absolute_position-Vector(0.0, self.h*0.5, 0.0))
                beta_k, beta = problem.evaluate_beta(absolute_position-Vector(0.0, 0.0, self.h*0.5))
                beta, beta_xyz = problem.evaluate_beta(absolute_position)

            u, u_xyz, u_xxyyzz = problem.evaluate_u(absolute_position)
            f = a * alpha * u - (
                b * (
                    (beta_xyz.i * u_xyz.i + beta_xyz.j * u_xyz.j + beta_xyz.k * u_xyz.k) +
                    beta * (u_xxyyzz.i + u_xxyyzz.j + u_xxyyzz.k)
                )
            )

            self.cell_values[element_index] = u
            self.true_solution[element_index] = f
            self.alpha[element_index] = alpha
            self.beta_face_values[SimpleLevel.FACE_I][element_index] = beta_i
            self.beta_face_values[SimpleLevel.FACE_J][element_index] = beta_j
            self.beta_face_values[SimpleLevel.FACE_K][element_index] = beta_k

    def print(self, title=None):
        if title:
            print(title)

        for i in range(self.space.i-1, -1, -1):
            for j in range(self.space.j-1, -1, -1):
                print(" "*j*4, end="")
                for k in range(self.space.k):
                    print("{:6.2f}".format(self.cell_values[(i, j, k)]), end="")
                print()
            print()
            print()


class SimpleMultigridSolver(object):
    def __init__(self, configuration, fine_level):
        if configuration.equation == 'h':
            self.a = 1.0
            self.b = 1.0
        else:
            self.a = 0.0
            self.b = 1.0

        self.interpolator = InterpolatorPC(pre_scale=0.0)
        self.restrictor = Restriction().restrict
        self.problem_operator = ConstantCoefficient7pt(self.a, self.b)

        if configuration.smoother == 'j':
            self.smoother = ShivJacobi(self.problem_operator, 6)
        else:
            raise Exception()

        self.bottom_solver = IdentityBottomSolver().solve

    def compute_residual(self):
        residual_operator = ConstantCoefficient7pt()

    def v_cycle(self, level):
        if min(level.space) < 3:
            self.bottom_solver_function(level)
            return

        level.cell_values = self.smoother(level.true_solution, level.cell_values)
        

        coarser_level = level.make_coarser_level()



        # do all restrictions

        self.restrict_function(coarser_level.cell_values, level.cell_values, Restriction.RESTRICT_CELL)
        self.restrict_function(coarser_level.beta_face_values[SimpleLevel.FACE_I],
                               level.beta_face_values[SimpleLevel.FACE_I], Restriction.RESTRICT_FACE_I)
        self.restrict_function(coarser_level.beta_face_values[SimpleLevel.FACE_J],
                               level.beta_face_values[SimpleLevel.FACE_J], Restriction.RESTRICT_FACE_J)
        self.restrict_function(coarser_level.beta_face_values[SimpleLevel.FACE_K],
                               level.beta_face_values[SimpleLevel.FACE_K], Restriction.RESTRICT_FACE_K)

        coarser_level.print("Coarsened level {}".format(coarser_level.level_number))
        for smooth_pass in range(self.smooth_iterations):
            new_cell_values = Mesh(coarser_level.space)
            self.smooth_function(new_cell_values, coarser_level.cell_values, 0.5, 1.0/12.0)
            coarser_level.cell_values = new_cell_values

        self.v_cycle(coarser_level)

        self.interpolate_function.interpolate(coarser_level.cell_values, level.cell_values)

        level.print("Interpolated level {}".format(level.level_number))


class IdentityBottomSolver(object):
    def solve(self, level):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log2_level_size', help='Size of space will be 3d, each dim will be of 2^(log2_level_size)',
                        default=6, type=int)
    parser.add_argument('-p', '--problem',
                        help="problem name, one of [sine]",
                        default='sine',
                        choices=['sine'], )
    parser.add_argument('-bc', '--boundary-conditions', dest='boundary_condition',
                        help="Type of boundary condition. Use p for Periodic and d for Dirichlet. Default is d",
                        default=('p' if os.environ.get('USE_PERIODIC_BC', 0) else 'd'),
                        choices=['p', 'd'], )
    parser.add_argument('-eq', '--equation',
                        help="Type of equation, h for helmholtz orp for poisson",
                        default='p' )
    parser.add_argument('-sm', '--smoother',
                        help="Type of smoother, j for jacobi",
                        default='j' )
    parser.add_argument('-fb', '--fixed_beta', dest='fixed_beta', action='store_true',
                        help="Use 1.0 as fixed value of beta, default is variable beta coefficient",
                        default=False, )
    command_line_configuration = parser.parse_args()

    global_size = Space([2**command_line_configuration.log2_level_size for _ in range(3)])

    fine_level = SimpleLevel(global_size, 0, command_line_configuration)
    fine_level.initialize()
    fine_level.print()

    solver = SimpleMultigridSolver(command_line_configuration, fine_level)
    solver.solve()