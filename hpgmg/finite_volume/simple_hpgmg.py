"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function
import argparse
import os
from hpgmg.finite_volume.Simple7PointOperator import SimpleConstantCoefficientOperator
from hpgmg.finite_volume.operators.interpolation import InterpolatorPC
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.operators.residual import Residual
from hpgmg.finite_volume.operators.restriction import Restriction

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg.finite_volume.space import Space, Coord
from hpgmg.finite_volume.simple_level import SimpleLevel


class SimpleMultigridSolver(object):
    """
    a simple multi-grid solver. gets a argparse configuration
    with settings from the command line
    """
    def __init__(self, configuration):
        if configuration.equation == 'h':
            # h is for helmholtz
            self.a = 1.0
            self.b = 1.0
        else:
            # default p is for poisson
            self.a = 0.0
            self.b = 1.0

        self.configuration = configuration
        self.dimensions = configuration.dimensions
        self.global_size = Space([2**configuration.log2_level_size for _ in range(self.dimensions)])
        self.ghost_zone = Coord(1 for _ in range(self.dimensions))

        self.number_of_v_cycles = configuration.number_of_vcycles
        self.interpolator = InterpolatorPC(pre_scale=0.0)
        self.restrictor = Restriction()
        self.problem_operator = SimpleConstantCoefficientOperator(solver=self)
        self.residual = Residual(solver=self)

        if configuration.smoother == 'j':
            self.smoother = JacobiSmoother(self.problem_operator, 6)
        else:
            raise Exception()

        self.bottom_solver = IdentityBottomSolver().solve

        self.fine_level = SimpleLevel(solver=self, space=self.global_size, level_number=0)
        self.fine_level.initialize()
        self.fine_level.print()

    def v_cycle(self, level):
        if min(level.space) < 3:
            self.bottom_solver(level)
            return

        self.smoother.smooth(level, level.cell_values, level.right_hand_side)
        self.residual.run(level, level.temp, level.cell_values, level.right_hand_side)

        coarser_level = level.make_coarser_level()
        self.problem_operator.rebuild_operator(coarser_level, level)
        self.restrictor.restrict(coarser_level, coarser_level.right_hand_side, level.temp, Restriction.RESTRICT_CELL)

        coarser_level.print("Coarsened level {}".format(coarser_level.level_number))



        # self.smoother.smooth(coarser_level, coarser_level.cell_values, coarser_level.exact_solution)

        self.v_cycle(coarser_level)

        self.interpolator.interpolate(coarser_level.cell_values, level.cell_values)

        level.print("Interpolated level {}".format(level.level_number))

    def solve(self):
        """
        MGSolve(&all_grids,VECTOR_U,VECTOR_F,a,b,dtol,rtol);
        void MGSolve(mg_type *all_grids, int u_id, int F_id, double a, double b, double dtol, double rtol){

            MGVCycle(all_grids,e_id,R_id,a,b,level);
        :return:
        """
        for cycle in range(self.number_of_v_cycles):
            print("Running v-cycle {}".format(cycle))
            self.v_cycle(self.fine_level)

    @staticmethod
    def get_configuration(args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('log2_level_size', help='each dim will be of 2^(log2_level_size)',
                            default=6, type=int)
        parser.add_argument('-d', '--dimensions', help='number of dimensions in problem',
                            default=3, type=int)
        parser.add_argument('-nv', '--number-of-vcycles', help='number of vcycles to run',
                            default=1, type=int)
        parser.add_argument('-gs', '--ghost_size', help='size of ghost zone (assumed symmetric)',
                            default=1, type=int)
        parser.add_argument('-p', '--problem',
                            help="problem name, one of [sine]",
                            default='sine',
                            choices=['sine'], )
        parser.add_argument('-bc', '--boundary-condition',
                            help="Type of boundary condition. Use p for Periodic and d for Dirichlet. Default is d",
                            default=('p' if os.environ.get('USE_PERIODIC_BC', 0) else 'd'),
                            choices=['p', 'd'], )
        parser.add_argument('-eq', '--equation',
                            help="Type of equation, h for helmholtz orp for poisson",
                            default='p', )
        parser.add_argument('-sm', '--smoother',
                            help="Type of smoother, j for jacobi",
                            default='j', )
        parser.add_argument('-fb', '--fixed_beta', action='store_true',
                            help="Use 1.0 as fixed value of beta, default is variable beta coefficient",
                            default=False, )
        return parser.parse_args(args=args)

    @staticmethod
    def get_solver(args=None):
        """
        this is meant for use in testing
        :return:
        """
        return SimpleMultigridSolver(SimpleMultigridSolver.get_configuration(args=args))

    @staticmethod
    def main():
        configuration = SimpleMultigridSolver.get_configuration()
        solver = SimpleMultigridSolver(configuration)
        solver.solve()


class IdentityBottomSolver(object):
    def solve(self, level):
        pass


if __name__ == '__main__':
    SimpleMultigridSolver.main()