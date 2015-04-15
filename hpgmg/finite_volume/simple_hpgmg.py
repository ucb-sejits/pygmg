"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function
import argparse
import os
from hpgmg.finite_volume.Simple7PointOperator import SimpleConstantCoefficientOperator
from hpgmg.finite_volume.operators.interpolation import InterpolatorPC
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother, ShivJacobi
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.operators.stencil_operators import ConstantCoefficient7pt

__author__ = 'nzhang-dev'

from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.operators.problem_sine import SineProblem
from hpgmg.finite_volume.smoothers import gauss_siedel, jacobi_stencil


class SimpleMultigridSolver(object):
    """
    a simple multi-grid solver. gets a argparse configuration
    with settings from the command line
    """
    def __init__(self, configuration):
        if configuration.equation == 'h':
            self.a = 1.0
            self.b = 1.0
        else:
            self.a = 0.0
            self.b = 1.0

        self.dimensions = configuration.dimensions
        self.global_size = Space([2**configuration.log2_level_size for _ in range(self.dimensions)])

        self.number_of_v_cycles = configuration.number_of_vcycles
        self.interpolator = InterpolatorPC(pre_scale=0.0)
        self.restrictor = Restriction().restrict
        self.problem_operator = SimpleConstantCoefficientOperator(solver=self)

        if configuration.smoother == 'j':
            self.smoother = ShivJacobi(self.problem_operator, 6)
        else:
            raise Exception()

        self.bottom_solver = IdentityBottomSolver().solve

        self.fine_level = SimpleLevel(self.global_size, 0, configuration)
        self.fine_level.initialize()
        self.fine_level.print()


    def compute_residual(self):
        residual_operator = ConstantCoefficient7pt()

    def v_cycle(self, level):
        if min(level.space) < 3:
            self.bottom_solver_function(level)
            return

        self.problem_operator.set_scale(level.h)

        level.cell_values = self.smoother.smooth(level.true_solution, level.cell_values)

        coarser_level = level.make_coarser_level()

        # do all restrictions

        coarser_level.print("Coarsened level {}".format(coarser_level.level_number))
        for smooth_pass in range(self.smooth_iterations):
            new_cell_values = Mesh(coarser_level.space)
            self.smooth_function(new_cell_values, coarser_level.cell_values, 0.5, 1.0/12.0)
            coarser_level.cell_values = new_cell_values

        self.v_cycle(coarser_level)

        self.interpolate_function.interpolate(coarser_level.cell_values, level.cell_values)

        level.print("Interpolated level {}".format(level.level_number))

    def solve(self):
        for cycle in range(self.number_of_v_cycles):
            print("Running v-cycle {}".format(cycle))
            self.v_cycle(self.fine_level)

    @staticmethod
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dimensions', help='number of dimensions in problem',
                            default=3, type=int)
        parser.add_argument('-nv', '--number-of-vcycles', help='number of vcycles to run',
                            default=1, type=int)
        parser.add_argument('log2_level_size', help='each dim will be of 2^(log2_level_size)',
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

        solver = SimpleMultigridSolver(command_line_configuration)
        solver.solve()


class IdentityBottomSolver(object):
    def solve(self, level):
        pass


if __name__ == '__main__':
    SimpleMultigridSolver.main()