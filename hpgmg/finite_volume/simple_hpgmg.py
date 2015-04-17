"""
implement a simple single threaded, gmg solver
"""
from __future__ import division, print_function
import argparse
import os
from hpgmg.finite_volume.Simple7PointOperator import SimpleConstantCoefficientOperator
from hpgmg.finite_volume.iterative_solver import IterativeSolver
from hpgmg.finite_volume.operators.interpolation import InterpolatorPC
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.operators.residual import Residual
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.timer import Timer

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

        self.boundary_is_periodic = configuration.boundary_condition == 'p'
        self.boundary_is_dirichlet = configuration.boundary_condition != 'p'

        if configuration.smoother == 'j':
            self.smoother = JacobiSmoother(self.problem_operator, 6)
        else:
            raise Exception()

        self.default_bottom_norm = 1e-3
        self.bottom_solver = IterativeSolver(solver=self, desired_reduction=self.default_bottom_norm)

        self.fine_level = SimpleLevel(solver=self, space=self.global_size, level_number=0)
        self.fine_level.initialize()
        self.fine_level.print()

    def v_cycle(self, level, target_mesh, residual_mesh):
        if min(level.space) <= 3:
            with Timer('bottom_solve'):
                self.bottom_solver.solve(level, target_mesh, residual_mesh)
            return

        with Timer("v-cycle level {}".format(level.level_number)):
            self.smoother.smooth(level, level.cell_values, level.right_hand_side)
            self.residual.run(level, level.temp, level.cell_values, level.right_hand_side)

            coarser_level = level.make_coarser_level()
            self.problem_operator.rebuild_operator(coarser_level, level)
            self.restrictor.restrict(coarser_level, coarser_level.right_hand_side,
                                     level.temp, Restriction.RESTRICT_CELL)

            # coarser_level.print("Coarsened level {}".format(coarser_level.level_number))

        self.v_cycle(coarser_level, coarser_level.cell_values, coarser_level.residual)

        with Timer("v-cycle level {}".format(level.level_number)):
            self.interpolator.interpolate(coarser_level.cell_values, level.cell_values)

        # level.print("Interpolated level {}".format(level.level_number))

    def solve(self):
        """
        void MGVCycle(mg_type *all_grids, int e_id, int R_id, double a, double b, int level){
        void MGSolve(mg_type *all_grids, int u_id, int F_id, double a, double b, double dtol, double rtol){

        MGSolve(&all_grids,VECTOR_U,VECTOR_F,a,b,dtol,rtol);
            MGVCycle(all_grids,e_id,R_id,a,b,level);
        :return:
        """
        d_tolerance, r_tolerance = 0.0, 1e-10
        norm_of_right_hand_side = 1.0
        norm_of_d_right_hand_side = 1.0

        level = self.fine_level

        if d_tolerance > 0.0:
            level.multiply_meshes(level.temp, 1.0, level.right_hand_side, level.d_inverse)
            norm_of_d_right_hand_side = level.norm_mesh(level.temp)
        if r_tolerance > 0.0:
            norm_of_right_hand_side = level.norm_mesh(level.right_hand_side)

        level.fill_mesh(level.cell_values, 0.0)
        level.scale_mesh(level.residual, 1.0, level.right_hand_side)

        for cycle in range(self.number_of_v_cycles):
            print("Running v-cycle {}".format(cycle))
            self.v_cycle(level, level.cell_values, level.residual)

            if self.boundary_is_periodic and self.a == 0.0 or level.alpha_is_zero:
                average_value_of_u = level.mean_mesh(level.cell_values)
                level.shift_mesh(level.cell_values, level.cell_values, average_value_of_u)

            self.residual.run(level, level.temp, level.cell_values, level.right_hand_side,)
            if d_tolerance > 0.0:
                level.multiply_meshes(level.temp, 1.0, level.temp, level.d_inverse)
            norm_of_residual = level.norm_mesh(level.temp)

            if cycle > 0:
                print("\n           ", end="")
                if r_tolerance > 0:
                    print("v-cycle={:2d}  norm={:1.15e}  rel={:%1.15e}".format(
                        cycle+1, norm_of_residual, norm_of_residual / norm_of_right_hand_side))
                else:
                    print("v-cycle={:2d}  norm={:1.15e}  rel={:1.15e}".format(
                        cycle+1, norm_of_residual, norm_of_residual / norm_of_d_right_hand_side))

            if norm_of_residual / norm_of_right_hand_side < r_tolerance:
                break
            if norm_of_residual < d_tolerance:
                break

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
        Timer.show_timers()


class IdentityBottomSolver(object):
    def solve(self, level):
        pass


if __name__ == '__main__':
    SimpleMultigridSolver.main()