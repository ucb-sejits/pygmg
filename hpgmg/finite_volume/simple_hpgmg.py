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
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.operators.problem_sine import SineProblem
from hpgmg.finite_volume.smoothers import gauss_siedel, jacobi_stencil


class SimpleMultigridSolver(object):
    def __init__(self, configuration):
        if configuration.equation == 'h':
            self.a = 1.0
            self.b = 1.0
        else:
            self.a = 0.0
            self.b = 1.0

        self.dimensions = configuration.dimensions
        self.global_size = Space([2**configuration.log2_level_size for _ in range(self.dimensions)])

        self.interpolator = InterpolatorPC(pre_scale=0.0)
        self.restrictor = Restriction().restrict
        self.problem_operator = ConstantCoefficient7pt(self.a, self.b)

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

    def solve(self):
        pass

    @staticmethod
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dimensions', help='number of dimensions in problem',
                            default=3, type=int)
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