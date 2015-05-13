"""
implement a simple single threaded, gmg solver
"""
from __future__ import division, print_function
import argparse
import os
import logging
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.chebyshev_smoother import ChebyshevSmoother

from hpgmg.finite_volume.operators.stencil_von_neumann_r1 import StencilVonNeumannR1
from hpgmg.finite_volume.operators.interpolation import InterpolatorPC
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.problems.problem_p4 import ProblemP4
from hpgmg.finite_volume.problems.problem_p6 import ProblemP6
from hpgmg.finite_volume.problems.problem_sine_n_dim import SineProblemND
from hpgmg.finite_volume.operators.residual import Residual
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.operators.variable_beta_generators import VariableBeta
from hpgmg.finite_volume.operators.boundary_conditions_fv import BoundaryUpdaterV1
from hpgmg.finite_volume.solvers.bicgstab import BiCGStab
from hpgmg.finite_volume.solvers.smoothing_solver import SmoothingSolver
from hpgmg.finite_volume.timer import Timer
from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.simple_level import SimpleLevel

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class SimpleMultigridSolver(object):
    """
    a simple multi-grid solver. gets a argparse configuration
    with settings from the command line
    """
    def __init__(self, configuration):
        self.dump_grids = configuration.dump_grids
        if self.dump_grids:
            Mesh.dump_mesh_enabled = True

        if configuration.log:
            logging.basicConfig(level=logging.DEBUG)

        logging.debug("equation {}".format(configuration.equation))

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

        self.is_variable_coefficient = configuration.variable_coefficient

        self.boundary_is_periodic = configuration.boundary_condition == 'p'
        self.boundary_is_dirichlet = configuration.boundary_condition != 'p'
        self.boundary_updater = BoundaryUpdaterV1(solver=self)

        self.is_helmholtz = configuration.equation == 'h'
        self.is_poisson = configuration.equation == 'p'

        self.number_of_v_cycles = configuration.number_of_vcycles
        self.interpolator = InterpolatorPC(solver=self, pre_scale=1.0)
        self.restrictor = Restriction(solver=self)

        self.problem_operator = StencilVonNeumannR1(solver=self)
        self.ghost_zone = self.problem_operator.ghost_zone

        self.residual = Residual(solver=self)
        if configuration.smoother == 'j':
            self.smoother = JacobiSmoother(
                self.problem_operator,
                use_l1_jacobi=configuration.use_l1_jacobi,
                iterations=configuration.smoother_iterations
            )
        elif configuration.smoother == 'c':
            self.smoother = ChebyshevSmoother(
                self.problem_operator,
                degree=1,
                iterations=configuration.smoother_iterations
            )

        self.default_bottom_norm = 1e-3
        if configuration.bottom_solver == 'bicgstab':
            self.bottom_solver = BiCGStab(solver=self, desired_reduction=self.default_bottom_norm)
        else:
            self.bottom_solver = SmoothingSolver(solver=self, desired_reduction=self.default_bottom_norm)

        self.minimum_coarse_dimension = configuration.minimum_coarse_dimension

        if configuration.problem_name == 'sine':
            self.problem = SineProblemND(dimensions=self.dimensions)
        elif configuration.problem_name == 'p4':
            self.problem = ProblemP4(dimensions=self.dimensions)
        elif configuration.problem_name == 'p6':
            self.problem = ProblemP6(
                dimensions=self.dimensions,
                shift=0.0 if not self.boundary_is_periodic else 1.0/21.0
            )

        if configuration.variable_coefficient:
            self.beta_generator = VariableBeta(self.dimensions)

        self.fine_level = SimpleLevel(solver=self, space=self.global_size, level_number=0)
        self.all_levels = [self.fine_level]

        self.initialize(self.fine_level)

        self.fine_level.exact_solution.dump("VECTOR_UTRUE")
        self.fine_level.right_hand_side.dump("VECTOR_F")

        if (self.a == 0.0 or self.fine_level.alpha_is_zero) and self.boundary_is_periodic:
            # Poisson w/ periodic BC's...
            # nominally, u shifted by any constant is still a valid solution.
            # However, by convention, we assume u sums to zero.
            average_value_of_exact_solution = self.fine_level.mean_mesh(self.fine_level.exact_solution)
            print("  average value of u_true = {:20.12e}... shifting u_true to ensure it sums to zero...".format(
                average_value_of_exact_solution
            ))
            self.fine_level.shift_mesh(self.fine_level.exact_solution, -average_value_of_exact_solution,
                                       self.fine_level.exact_solution)
            self.fine_level.dump(self.fine_level.exact_solution, "VECTOR_UTRUE_ADJUSTED")

        if self.boundary_is_periodic:
            average_value_of_rhs = self.fine_level.mean_mesh(self.fine_level.right_hand_side)
            if average_value_of_rhs != 0.0:
                print("average_value_of_rhs {} should be zero, adjusting it".format(average_value_of_rhs))
                self.fine_level.shift_mesh(
                    self.fine_level.right_hand_side,
                    average_value_of_rhs,
                    self.fine_level.right_hand_side
                )
                self.fine_level.right_hand_side.dump("VECTOR_F_ADJUSTED")

        self.problem_operator.rebuild_operator(self.fine_level, source_level=None)

        if self.dimensions == 3:
            self.fine_level.beta_face_values[0].dump("VECTOR_BETA_K")
            self.fine_level.beta_face_values[1].dump("VECTOR_BETA_J")
            self.fine_level.beta_face_values[2].dump("VECTOR_BETA_I")

    def initialize(self, level):
        """
        Initialize the right_hand_side(VECTOR_F), exact_solution(VECTOR_UTRUE)
        alpha, and possibly the beta_faces
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0
        beta_xyz = Vector(0.0, 0.0, 0.0)
        face_betas = [1.0 for _ in range(self.dimensions)]

        problem = self.problem
        if level.is_variable_coefficient:
            beta_generator = self.beta_generator

        for element_index in level.indices():
            absolute_position = level.coord_to_cell_center_point(element_index)

            if level.is_variable_coefficient:
                for face_index in range(self.dimensions):
                    # print("##,{} ".format(",".join(map(str, element_index))), end="")
                    # the following face_betas are reversed in order to keep
                    # strict compatibility with c,
                    # TODO: figure out if there is a way to get iteration to do this naturally
                    # face_betas[self.dimensions-(face_index+1)], _ = beta_generator.evaluate_beta(
                    #     level.coord_to_face_center_point(element_index, face_index)
                    # )
                    face_betas[face_index], _ = beta_generator.evaluate_beta(
                        level.coord_to_face_center_point(element_index, face_index)
                    )
                    # face_betas[face_index] = face_index * 1000.0 + element_index[2] * 100 + \
                    #     element_index[1] * 10 + element_index[0]

                # print("{}".format(",".join(map(str, element_index))), end="")
                beta, beta_xyz = beta_generator.evaluate_beta(absolute_position)

            u, u_xyz, u_xxyyzz = problem.evaluate_u(absolute_position)

            # double F = a*A*U - b*( (Bx*Ux + By*Uy + Bz*Uz)  +  B*(Uxx + Uyy + Uzz) );
            f = self.a * alpha * u - (
                self.b * ((beta_xyz * u_xyz) + beta * sum(u_xxyyzz))
            )

            # print("init {:12s} {:20} u {:e} beta_xyz ({}) u_xyz {} u_xxyyzz {} f {:8.6f}".format(
            #     element_index, absolute_position, u,
            #     ",".join("{:8.6f}".format(n) for n in beta_xyz),
            #     ",".join("{:8.6f}".format(n) for n in u_xyz),
            #     ",".join("{:8.6f}".format(n) for n in u_xxyyzz), f
            # ))

            level.right_hand_side[element_index] = f
            level.exact_solution[element_index] = u

            level.alpha[element_index] = alpha
            for face_index in range(self.dimensions):
                # if all(element_index[d] < level.space[d]-1 for d in range(self.dimensions)):
                level.beta_face_values[face_index][element_index] = face_betas[face_index]

        if level.alpha_is_zero is None:
            level.alpha_is_zero = level.dot_mesh(level.alpha, level.alpha) == 0.0
        logging.debug("level.alpha_is_zero {}".format(level.alpha_is_zero))

    def build_all_levels(self):
        level = self.fine_level
        while level.space[0] > self.minimum_coarse_dimension and level.space[0] % 2 == 0:
            coarser_level = level.make_coarser_level()
            self.all_levels.append(coarser_level)
            level = coarser_level

    def v_cycle(self, level, target_mesh, residual_mesh):
        if min(level.space) <= 3:
            with level.timer('total cycles'):
                residual_mesh.dump("BOTTOM-SOLVER-RESIDUAL level {}".format(level.level_number))
                self.bottom_solver.solve(level, target_mesh, residual_mesh)
                target_mesh.dump("BOTTOM-SOLVED level {}".format(level.level_number))
            return

        level.right_hand_side.dump("VCYCLE_RHS")
        with level.timer("total cycles"):
            self.smoother.smooth(level, level.cell_values, level.residual)
            # if level.level_number == 1:
            #     exit(0)
            level.cell_values.dump("PRE-SMOOTH VECTOR_U level {}".format(level.level_number))
            self.residual.run(level, level.temp, level.cell_values, level.right_hand_side)

            level.temp.dump("VECTOR_TEMP_RESIDUAL level {}".format(level.level_number))

            coarser_level = level.make_coarser_level()

            self.restrictor.restrict(coarser_level, coarser_level.residual, level.temp, Restriction.RESTRICT_CELL)
            coarser_level.residual.dump("RESTRICTED_RID level {}".format(coarser_level.level_number))
            coarser_level.fill_mesh(coarser_level.cell_values, 0.0)

            self.problem_operator.rebuild_operator(coarser_level, level)
            coarser_level.residual.dump("RESTRICTED_RID level {}".format(coarser_level.level_number))

        self.v_cycle(coarser_level, coarser_level.cell_values, coarser_level.residual)

        with level.timer("total cycles"):
            self.interpolator.interpolate(level, level.cell_values, coarser_level.cell_values)
            level.cell_values.dump("INTERPOLATED_U level {}".format(level.level_number))
            self.smoother.smooth(level, level.cell_values, level.residual)
            level.cell_values.dump("POST-SMOOTH VECTOR_U level {}".format(level.level_number))

        # level.print("Interpolated level {}".format(level.level_number))

    def solve(self):
        """
        void MGSolve(mg_type *all_grids, int u_id, int F_id, double a, double b, double dtol, double rtol){
            void MGVCycle(mg_type *all_grids, int e_id, int R_id, double a, double b, int level){

        MGSolve(&all_grids,VECTOR_U,VECTOR_F,a,b,dtol,rtol);
            MGVCycle(all_grids,e_id,R_id,a,b,level);


        MGSolve(&all_grids, u_id=VECTOR_U(cell_values), F_id=VECTOR_F(right_hand_side), a, b, dtol,rtol);
            e_id = u_id(cell_values)
            R_id = VECTOR_F_MINUS_FV(residual)
            MGVCycle(all_grids, e_id=VECTOR_U(cell_values), R_id=VECTOR_F_MINUS_FV(residual),a,b,level);
        :return:
        """
        d_tolerance, r_tolerance = 0.0, 1.0e-10
        norm_of_right_hand_side = 1.0
        norm_of_d_right_hand_side = 1.0

        level = self.fine_level

        with Timer("mg-solve time"):
            print("MGSolve.... \n", end='')

            if d_tolerance > 0.0:
                level.multiply_meshes(level.temp, 1.0, level.right_hand_side, level.d_inverse)
                norm_of_d_right_hand_side = level.norm_mesh(level.temp)
            if r_tolerance > 0.0:
                norm_of_right_hand_side = level.norm_mesh(level.right_hand_side)

            level.fill_mesh(level.cell_values, 0.0)
            level.scale_mesh(level.residual, 1.0, level.right_hand_side)

            level.residual.dump('VECTOR_F_MINUS_AV')

            for cycle in range(self.number_of_v_cycles):
                # level.residual.print('residual before v_cycle')
                self.v_cycle(level, level.cell_values, level.residual)
                # exit(0)
                level.cell_values.dump('cell values after v_cycle')

                if self.boundary_is_periodic and self.a == 0.0 or level.alpha_is_zero:
                    # Poisson with Periodic Boundary Conditions...
                    # by convention, we assume the solution sums to zero...
                    # so eliminate any constants from the solution...
                    average_value_of_u = level.mean_mesh(level.cell_values)
                    level.shift_mesh(level.cell_values, -average_value_of_u, level.cell_values)

                level.right_hand_side.dump("right hand side after v_cycle")
                self.residual.run(level, level.temp, level.cell_values, level.right_hand_side,)
                level.temp.dump("residual after v_cycle")
                if d_tolerance > 0.0:
                    level.multiply_meshes(level.temp, 1.0, level.temp, level.d_inverse)
                norm_of_residual = level.norm_mesh(level.temp)

                if cycle > 0:
                    print("\n           ", end="")
                if r_tolerance > 0:
                    print("v-cycle={:2d}  norm={:1.15e}  rel={:1.15e}".format(
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
        parser.add_argument('-pn', '--problem-name',
                            help="math problem name to use for initialization",
                            default='sine',
                            choices=['sine', 'p4', 'p6'], )
        parser.add_argument('-bc', '--boundary-condition',
                            help="Type of boundary condition. Use p for Periodic and d for Dirichlet. Default is d",
                            default=('p' if os.environ.get('USE_PERIODIC_BC', 0) else 'd'),
                            choices=['p', 'd'], )
        parser.add_argument('-eq', '--equation',
                            help="Type of equation, h for helmholtz or p for poisson",
                            choices=['h', 'p'],
                            default='h', )
        parser.add_argument('-sm', '--smoother',
                            help="Type of smoother, j for jacobi, c for chebyshev",
                            choices=['j', 'c'],
                            default='j', )
        parser.add_argument('-bs', '--bottom-solver',
                            help="Bottom solver to use",
                            choices=['smoother', 'bicgstab'],
                            default='bicgstab', )
        parser.add_argument('-ulj', '--use-l1-jacobi', action="store_true",
                            help="use l1 instead of d inverse with jacobi smoother",
                            default=False, )
        parser.add_argument('-vc', '--variable-coefficient', action='store_true',
                            help="Use 1.0 as fixed value of beta, default is variable beta coefficient",
                            default=False, )
        parser.add_argument('-si', '--smoother-iterations', help='number of iterations each time smoother is called',
                            default=6, type=int)
        parser.add_argument('-mcd', '--minimum-coarse_dimension', help='smallest allowed coarsened dimension',
                            default=3, type=int)
        parser.add_argument('-dg', '--dump-grids', help='dump various grids for comparison with hpgmg.c',
                            action="store_true", default=False)
        parser.add_argument('-l', '--log', help='turn on logging', action="store_true", default=False)
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


if __name__ == '__main__':
    SimpleMultigridSolver.main()