"""
implement a simple single threaded, gmg solver
"""
from __future__ import division, print_function
import argparse
import functools
import os
import sys
import logging
import time
from snowflake.stencil_compiler import CCompiler
from snowflake_openmp.compiler import TiledOpenMPCompiler, OpenMPCompiler
import sympy
from hpgmg import finite_volume
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.chebyshev_smoother import ChebyshevSmoother
from hpgmg.finite_volume.operators.specializers.initialize_mesh_specializer import CInitializeMesh, OclInitializeMesh
from hpgmg.finite_volume.operators.specializers.util import profile, time_this, specialized_func_dispatcher

from hpgmg.finite_volume.operators.stencil_von_neumann_r1 import StencilVonNeumannR1
from hpgmg.finite_volume.operators.interpolation import InterpolatorPC
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.problems.problem_fv import ProblemFV
from hpgmg.finite_volume.problems.problem_p4 import ProblemP4
from hpgmg.finite_volume.problems.problem_p6 import ProblemP6
from hpgmg.finite_volume.problems.problem_sine_n_dim import SineProblemND
from hpgmg.finite_volume.operators.residual import Residual
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.operators.variable_beta_generators import VariableBeta
from hpgmg.finite_volume.operators.boundary_conditions_fv import BoundaryUpdaterV1
from hpgmg.finite_volume.solvers.bicgstab import BiCGStab
from hpgmg.finite_volume.solvers.smoothing_solver import SmoothingSolver
from hpgmg.finite_volume.timer import EventTimer
from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.simple_level import SimpleLevel

import numpy as np
import pycl as cl
import ctypes

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class SimpleMultigridSolver(object):
    """
    a simple multi-grid solver. gets a argparse configuration
    with settings from the command line
    """
    @profile
    def __init__(self, configuration):
        self.compiler = CCompiler
        self.dump_grids = configuration.dump_grids
        if self.dump_grids:
            Mesh.dump_mesh_enabled = True

        if configuration.log:
            logging.basicConfig(level=logging.DEBUG)

        logging.debug("equation {}".format(configuration.equation))

        if configuration.equation == 'h':
            self.a = 1.0
            self.b = 1.0
            print("Creating Helmholtz (a={}, b={}) test problem".format(self.a, self.b))
        else:
            # default p is for poisson
            self.a = 0.0
            self.b = 1.0
            print("Creating Poisson(a={}, b={}) test problem".format(self.a, self.b))

        self.configuration = configuration
        self.dimensions = configuration.dimensions
        self.global_size = Space([2**configuration.log2_level_size for _ in range(self.dimensions)])

        self.is_variable_coefficient = configuration.variable_coefficient

        self.backend = self.configuration.backend

        self.boundary_is_periodic = configuration.boundary_condition == 'p'
        self.boundary_is_dirichlet = configuration.boundary_condition != 'p'
        self.boundary_updater = BoundaryUpdaterV1(solver=self)
        print("Boundary condition is {}".format(self.boundary_updater.name))
        self.is_helmholtz = configuration.equation == 'h'
        self.is_poisson = configuration.equation == 'p'

        self.number_of_v_cycles = configuration.number_of_vcycles
        self.interpolator = InterpolatorPC(solver=self, pre_scale=1.0)
        self.restrictor = Restriction(solver=self)

        self.do_f_cycle = configuration.do_f_cycles
        self.unlimited_fmg_cycles = configuration.unlimited_fmg_cycles
        if self.do_f_cycle:
            print("Running using f-cycles")
        else:
            print("Running without f-cycles")
        self.problem_operator = StencilVonNeumannR1(solver=self)
        self.ghost_zone = self.problem_operator.ghost_zone

        self.residual = Residual(solver=self)
        if configuration.smoother == 'j':
            self.smoother = JacobiSmoother(
                self.problem_operator,
                use_l1_jacobi=configuration.use_l1_jacobi,
                iterations=configuration.smoother_iterations
            )
            print("Using Jacobi smoother ({})".format(
                "l1_inverse" if configuration.use_l1_jacobi else "d_inverse"
            ))
        elif configuration.smoother == 'c':
            self.smoother = ChebyshevSmoother(
                self.problem_operator,
                degree=1,
                iterations=configuration.smoother_iterations
            )
            print("Using Chebyshev smoother")

        self.default_bottom_norm = 1e-3
        if configuration.bottom_solver == 'bicgstab':
            self.bottom_solver = BiCGStab(solver=self, desired_reduction=self.default_bottom_norm)
            print("Using BiCGStab bottom solver")
        else:
            self.bottom_solver = SmoothingSolver(solver=self, desired_reduction=self.default_bottom_norm)
            print("Using smoothing bottom solver")

        self.minimum_coarse_dimension = configuration.minimum_coarse_dimension

        self.fine_level = SimpleLevel(solver=self, space=self.global_size, level_number=0)
        self.all_levels = [self.fine_level]

        if configuration.problem_name == 'sine':
            self.problem = SineProblemND(dimensions=self.dimensions)
        elif configuration.problem_name == 'fv':
            self.problem = ProblemFV(dimensions=self.dimensions,
                                     cell_size=self.fine_level.cell_size,
                                     add_4th_order_correction=True)
        elif configuration.problem_name == 'p4':
            self.problem = ProblemP4(dimensions=self.dimensions)
        elif configuration.problem_name == 'p6':
            self.problem = ProblemP6(
                dimensions=self.dimensions,
                shift=0.0 if not self.boundary_is_periodic else 1.0/21.0
            )
        print("Using problem initializer {}".format(configuration.problem_name))

        self.compute_richardson_error = not configuration.disable_richardson_error

        if configuration.variable_coefficient:
            self.beta_generator = VariableBeta(self.dimensions)
        else:
            self.beta_generator = None

        self.timer = EventTimer(self)

        if self.configuration.backend == 'ocl':
            self.context = cl.clCreateContext(devices=[cl.clGetDeviceIDs()[-1]])
            self.queue = cl.clCreateCommandQueue(self.context)
            fill_source = '''
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            kernel void fill_buffer(__global double* mesh, double value) {
                mesh[get_global_id(0)] = value;
            }
            '''
            self.fill_kernel = cl.clCreateProgramWithSource(self.context, fill_source).build()['fill_buffer']
            self.fill_kernel.argtypes = (cl.cl_mem, ctypes.c_double)

        self.fine_level = SimpleLevel(solver=self, space=self.global_size, level_number=0)
        self.all_levels = [self.fine_level]
        # self.initialize(self.fine_level)
        self.problem.initialize_problem(self, self.fine_level)

        self.fine_level.exact_solution.dump("VECTOR_UTRUE")
        self.fine_level.right_hand_side.dump("VECTOR_F", force_dump=True)

        if self.dimensions == 3:
            self.fine_level.beta_face_values[2].dump("VECTOR_BETA_I")
            self.fine_level.beta_face_values[1].dump("VECTOR_BETA_J")
            self.fine_level.beta_face_values[0].dump("VECTOR_BETA_K")

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
            self.fine_level.exact_solution.dump("VECTOR_UTRUE_ADJUSTED")

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

        self.build_all_levels()

        if self.dimensions == 3:
            for index in range(1, len(self.all_levels)):
                self.all_levels[index].beta_face_values[2].dump("VECTOR_BETA_I_LEVEL_{}".format(index))
                self.all_levels[index].beta_face_values[1].dump("VECTOR_BETA_J_LEVEL_{}".format(index))
                self.all_levels[index].beta_face_values[0].dump("VECTOR_BETA_K_LEVEL_{}".format(index))
                self.all_levels[index].d_inverse.dump("VECTOR_DINV_LEVEL_{}".format(index))


    @specialized_func_dispatcher({
        'c': CInitializeMesh,
        'omp': CInitializeMesh,
        'ocl': OclInitializeMesh
    })
    def initialize_mesh(self, level, mesh, exp, coord_transform, dump=False):
        """
        DEPRECATED: TODO REMOVE THIS FUNCTION ASAP, replaced in AlgebraicProblem
        WARNING: This function does not handle variable coefficient shifts
        """
        func = self.problem.get_func(exp, self.problem.symbols)
        # print("expression {}".format(exp))
        for coord in level.interior_points():
            if dump:
                x, y, z = coord_transform(coord)
                f = func(*coord_transform(coord))
                print("Coordinate ({:12.10f},{:12.10f},{:12.10f}) -> {:10.6g}".format(x, y, z, f))
            mesh[coord] = func(*coord_transform(coord))

    @time_this
    @profile
    def initialize(self, level):
        """
        DEPRECATED: TODO REMOVE THIS FUNCTION ASAP, replaced in problem definitions
        Initialize the right_hand_side(VECTOR_F), exact_solution(VECTOR_UTRUE)
        alpha, and possibly the beta_faces
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0
        # beta_xyz = Vector(0.0, 0.0, 0.0)
        # face_betas = [1.0 for _ in range(self.dimensions)]

        problem = self.problem
        print(self.problem.expression)

        beta_generator = self.beta_generator

        a = time.time()
        #fill Alpha
        #level.alpha
        level.alpha.fill(alpha)

        #fill U
        self.initialize_mesh(level, level.exact_solution, problem.expression, level.coord_to_cell_center_point)
        # beta stuff
        if level.is_variable_coefficient:
            # beta_values = np.fromfunction(
            #     np.frompyfunc(
            #         lambda *vector: beta_generator.evaluate_beta(vector), self.dimensions, 1
            #     ),
            #     level.space,
            #     dtype=level.exact_solution.dtype
            # )
            beta_expression = beta_generator.get_beta_expression()

            # beta_vectors = np.fromfunction(
            #     np.frompyfunc(
            #         lambda *vector: beta_generator.evaluate_beta_vector(vector), self.dimensions, self.dimensions
            #     ),
            #     level.space,
            #     dtype=level.exact_solution.dtype
            # )
        else:
            beta_expression = beta
            # beta_values = np.full(level.space, beta)
            # beta_vectors = [
            #     np.full(level.space, 0)
            #     for i in range(self.dimensions)
            # ]
        #
        # level.right_hand_side -= self.b * (sum(A*B for A, B in zip(beta_vectors, first_derivatives)) +
        #                                    beta_values * second_derivatives)

        if self.is_variable_coefficient:
            for face_index in range(self.dimensions):
                level.beta_face_values[face_index][:] = np.fromfunction(
                    np.frompyfunc(
                        lambda *vector: beta_generator.evaluate_beta(
                            level.coord_to_face_center_point(vector, face_index)
                        ),
                        self.dimensions,
                        1
                    ),
                    level.space,
                    dtype=level.exact_solution.dtype
                )
        symbols = self.problem.symbols
        beta_first_derivative = [sympy.diff(beta_expression, sym) for sym in symbols]
        u_first_derivative = [sympy.diff(self.problem.expression, sym) for sym in symbols]
        bu_derivative_1 = sum(a * b for a, b in zip(beta_first_derivative, u_first_derivative))
        u_derivative_2 = [sympy.diff(problem.expression, sym, 2) for sym in symbols]
        f_exp = self.a * alpha * problem.expression - self.b * (bu_derivative_1 + beta_expression * sum(u_derivative_2))
        #F = a*A*U - b*( (Bx*Ux + By*Uy + Bz*Uz)  +  B*(Uxx + Uyy + Uzz) );
        #print(f_exp)
        # rhs = np.zeros_like(level.right_hand_side)
        self.initialize_mesh(level, level.right_hand_side, f_exp, level.coord_to_cell_center_point, dump=False)
        # level.right_hand_side.dump("VECTOR_F:right_hand_side", force_dump=True)
        # exit(0)
        #
        # print(rhs.ravel()[:10])
        # print(level.right_hand_side.ravel()[:10])

        b = time.time()


        if level.alpha_is_zero is None:
            level.alpha_is_zero = level.dot_mesh(level.alpha, level.alpha) == 0.0
        logging.debug("level.alpha_is_zero {}".format(level.alpha_is_zero))

    @time_this
    @profile
    def build_all_levels(self):
        level = self.fine_level
        while level.space[0] > self.minimum_coarse_dimension and level.space[0] % 2 == 0:
            coarser_level = level.make_coarser_level()
            self.problem_operator.rebuild_operator(coarser_level, level)
            self.all_levels.append(coarser_level)
            level = coarser_level
        for level in self.all_levels:
            self.smoother.get_kernel(level)

        for level in self.all_levels:
            level.must_subtract_mean = False
            alpha_is_zero = level.dot_mesh(level.alpha, level.alpha) == 0.0
            if self.boundary_is_periodic and (self.a == 0.0 or alpha_is_zero == 1.0):
                level.must_subtract_mean = True

    def v_cycle(self, level, target_mesh, residual_mesh):
        if min(level.space) <= 3:
            with level.timer('total cycles'):
                #residual_mesh.dump("BOTTOM-SOLVER-RESIDUAL level {}".format(level.level_number))
                self.bottom_solver.solve(level, target_mesh, residual_mesh)
                #target_mesh.dump("BOTTOM-SOLVED level {}".format(level.level_number))
            return

        #level.right_hand_side.dump("VCYCLE_RHS")
        with level.timer("total cycles"):
            self.smoother.smooth(level, level.cell_values, level.residual)
            #level.cell_values.dump("PRE-SMOOTH VECTOR_U level {}".format(level.level_number))
            self.residual.run(level, level.temp, level.cell_values, residual_mesh)

            #level.temp.dump("VECTOR_TEMP_RESIDUAL level {}".format(level.level_number))

            coarser_level = self.all_levels[level.level_number+1]
            # print('pre:', np.sum(abs(level.temp.ravel())))
            # print(level.temp[:4, :4, :4])
            self.restrictor.restrict(coarser_level, coarser_level.residual, level.temp, Restriction.RESTRICT_CELL)
            # print('post:', np.sum(abs(coarser_level.residual.ravel())))
            # print(coarser_level.residual[:4, :4, :4])
            coarser_level.fill_mesh(coarser_level.cell_values, 0.0)

            coarser_level.residual.dump("RESTRICTED_RID level {}".format(coarser_level.level_number))

        self.v_cycle(coarser_level, coarser_level.cell_values, coarser_level.residual)

        with level.timer("total cycles"):
            self.interpolator.interpolate(level, level.cell_values, coarser_level.cell_values)
            level.cell_values.dump("INTERPOLATED_U level {}".format(level.level_number))
            self.smoother.smooth(level, level.cell_values, level.residual)
            level.cell_values.dump("POST-SMOOTH VECTOR_U level {}".format(level.level_number))

        # level.print("Interpolated level {}".format(level.level_number))

    @time_this
    def solve(self, start_level=0):
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

        level = self.all_levels[start_level]

        print("MGSolve.... ")
        sys.stdout.flush()

        with self.timer("mg-solve time"):
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

                if r_tolerance > 0:
                    print("      v-cycle={:2d}  norm={:1.15e}  rel={:1.15e}".format(
                        cycle+1, norm_of_residual, norm_of_residual / norm_of_right_hand_side))
                else:
                    print("      v-cycle={:2d}  norm={:1.15e}  rel={:1.15e}".format(
                        cycle+1, norm_of_residual, norm_of_residual / norm_of_d_right_hand_side))
                sys.stdout.flush()

                if norm_of_residual / norm_of_right_hand_side < r_tolerance:
                    break
                if norm_of_residual < d_tolerance:
                    break

    @time_this
    def solve_with_f_cycle(self, start_level_number, d_tolerance, r_tolerance):
        max_v_cycles = 10 if self.unlimited_fmg_cycles else 0
        number_of_levels = len(self.all_levels)
        start_level = self.all_levels[start_level_number]

        # calculate norm of f...
        norm_of_f, norm_of_d_inv_f = 1.0, 1.0
        if d_tolerance > 0.0:
            # D^{-1}F
            start_level.multiply_meshes(start_level.temp, 1.0, start_level.right_hand_side, start_level.d_inverse)
            norm_of_d_inv_f = start_level.norm_mesh(start_level.temp)  # ||D^{-1}F||
        if r_tolerance:
            norm_of_f = start_level.norm_mesh(start_level.right_hand_side)  # ||F||
            start_level.right_hand_side.dump(
                "F_id:right_hand_side with ||right_hand_side|| == {}".format(norm_of_f), force_dump=True)

        # initialize the RHS for the f-cycle to f...
        start_level.scale_mesh(start_level.residual, 1.0, start_level.right_hand_side)

        # restrict RHS to bottom (coarsest grids)
        for restrict_level in range(start_level_number, len(self.all_levels)-1):
            self.restrictor.restrict(self.all_levels[restrict_level+1],
                                     self.all_levels[restrict_level+1].residual,
                                     self.all_levels[restrict_level].residual, Restriction.RESTRICT_CELL)

        # solve coarsest grid...
        last_level_number = number_of_levels - 1
        last_level = self.all_levels[last_level_number]
        if last_level_number > start_level_number:
            start_level.fill_mesh(start_level.cell_values, 0.0)  # use whatever was the initial guess
            self.bottom_solver.solve(last_level, start_level.cell_values, start_level.residual)

        # now do the F-cycle proper...
        for cycle_level in range(number_of_levels-2, start_level_number-1, -1):
            # high-order interpolation
            source_coarse_level = self.all_levels[cycle_level+1]
            target_fine_level = self.all_levels[cycle_level]
            self.interpolator.interpolate(target_fine_level, target_fine_level.cell_values,
                                          source_coarse_level.cell_values)
            target_fine_level.v_cycles_from_this_level += 1
            self.v_cycle(target_fine_level, target_fine_level.cell_values, target_fine_level.residual)

        # now do the post-F V-cycles
        for v in range(-1, max_v_cycles):
            level_number = start_level_number

            # do the v-cycle...
            if v >= 0:
                self.all_levels[level_number].v_cycles_from_this_level += 1
                self.v_cycle(self.fine_level, self.fine_level.cell_values,
                             self.fine_level.residual)

            # now calculate the norm of the residual...
            if start_level.must_subtract_mean:
                average_value_of_cell_values = start_level.mean_mesh(start_level.cell_values)
                start_level.shift_mesh(start_level.cell_values, average_value_of_cell_values, start_level.cell_values)

            self.residual.run(start_level,
                              target_mesh=start_level.temp,
                              source_mesh=start_level.cell_values,
                              right_hand_side=start_level.right_hand_side)

            if d_tolerance > 0.0:
                start_level.multiply_meshes(start_level.temp, 1.0, start_level.temp, start_level.d_inverse)

            norm_of_residual = start_level.norm_mesh(start_level.temp)

            if r_tolerance > 0.0:
                rel = norm_of_residual / norm_of_f
            else:
                rel = norm_of_residual / norm_of_d_inv_f

            if v > 0.0:
                print("            v-cycle={:2d}  norm={:1.15e}  rel={:1.15e}  ".format(v+1, norm_of_residual, rel))
            else:
                print("            f-cycle={:2s}  norm={:1.15e}  rel={:1.15e}  ".format("", norm_of_residual, rel))

            if norm_of_residual / norm_of_f < r_tolerance:
                break
            if norm_of_residual < d_tolerance:
                break

    def richardson_error(self, start_level=0):
        import math
        """
        in FV...
        +-------+   +---+---+   +-------+   +-------+
        |       |   | a | b |   |       |   |a+b+c+d|
        |  u^2h | - +---+---+ = |  u^2h | - |  ---  |
        |       |   | c | d |   |       |   |   4   |
        +-------+   +---+---+   +-------+   +-------+
        """
        print("performing Richardson error analysis...")
        level_0 = self.all_levels[start_level]
        level_1 = self.all_levels[start_level+1]
        level_2 = self.all_levels[start_level+2]

        self.restrictor.restrict(level_1, level_1.temp, level_0.cell_values, Restriction.RESTRICT_CELL)
        self.restrictor.restrict(level_2, level_2.temp, level_1.cell_values, Restriction.RESTRICT_CELL)

        level_1.add_meshes(level_1.temp, 1.0, level_1.cell_values, -1.0, level_1.temp)
        level_2.add_meshes(level_2.temp, 1.0, level_2.cell_values, -1.0, level_2.temp)

        norm_of_u2h_minus_uh = level_1.norm_mesh(level_1.temp)  # || u^2h - R u^h  ||max
        norm_of_u4h_minus_u2h = level_2.norm_mesh(level_2.temp)  # || u^4h = R u^2h ||max
        # estimate the error^h using ||u^2h - R u^h||
        print("  h = {:22.15e}  ||error|| = {:22.15e}".format(level_0.h, norm_of_u2h_minus_uh))
        # log( ||u^4h - R u^2h|| / ||u^2h - R u^h|| ) / log(2)
        # is an estimate of the order of the method (e.g. 4th order)
        print("  order = {:0.10f}".format(math.log(norm_of_u4h_minus_u2h / norm_of_u2h_minus_uh) / math.log(2.0)))

    @time_this
    def run_richardson_test(self):
        if len(self.all_levels) < 3:
            print("WARNING: not enough levels to perform richardson test, skipping...")
            return

        for level_number in range(3):
            level = self.all_levels[level_number]
            if level_number > 0:
                self.restrictor.restrict(
                    level,
                    level.right_hand_side, self.all_levels[level_number-1].right_hand_side,
                    Restriction.RESTRICT_CELL)
            level.fill_mesh(level.cell_values, 0.0)

            if self.do_f_cycle:
                self.solve_with_f_cycle(0, 0.0, 1.0e-10)
            else:
                self.solve(start_level=level_number)
        self.richardson_error()

    def calculate_error(self, mesh1, mesh2):
        """
        returns max norm of the error function
        Commented code below is alternative that returns normalized l2 error
        :param mesh1:
        :param mesh2:
        :return:
        """
        level = self.fine_level
        level.add_meshes(level.temp, 1.0, mesh1, -1.0, mesh2)
        level.temp.dump("ERROR DIFFERENCE")
        error_norm = level.norm_mesh(level.temp)
        return error_norm

        # h3 = level.h ** self.dimensions
        # l2 = math.sqrt(level.dot_meshes(level.temp, level.temp)*h3)
        # return l2

    def show_error_information(self):
        level = self.fine_level
        level.exact_solution.dump("FINAL_EXACT_SOLUTION")
        level.cell_values.dump("FINAL_COMPUTED_SOLUTION")

        fine_error = self.calculate_error(self.fine_level.cell_values, self.fine_level.exact_solution)
        print("\ncalculating error... h = {:22.15e}  ||error|| = {:22.15e}".format(
            1.0/self.fine_level.dimension_size(), fine_error
        ))

    def show_timing_information(self):
        all_level_keys = set()
        for level in self.all_levels:
            for key in level.timer.names():
                all_level_keys.add(key)

        print("{:26.26s}".format(""), end=" ")
        for level in self.all_levels:
            print("{:12d}".format(level.level_number), end=" ")
        print()
        print("{:26.26s}".format("box dimension"), end=" ")
        for level in self.all_levels:
            print("{:>12s}".format("{}^{}".format(level.dimension_size(), self.dimensions)), end=" ")
        print(" {:>12s}".format("total"))
        print("{:26.26s}".format("-"*26), end=" ")
        for _ in self.all_levels:
            print("{:>12s}".format("-"*12), end=" ")
        print(" {:>12s}".format("-"*12))
        for key in sorted(all_level_keys):
            print("{:26.26s}".format(key), end=" ")
            row_total = 0.0
            for level in self.all_levels:
                if key in level.timer.names():
                    print("{:12.6f}".format(level.timer[key].total_time), end=" ")
                    row_total += level.timer[key].total_time
                else:
                    print("{:12s}".format("NA"), end=" ")
            print(" {:12f}".format(row_total))

        print()
        for key in self.timer.names():
            print(self.timer[key])

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
        parser.add_argument('-dre', '--disable-richardson-error',
                            help="don't compute or show richardson error at end of run",
                            action="store_true", default=False)
        parser.add_argument('-dfc', '--do-f-cycles',
                            help="use f-cycle solver instead of full v-cycle solver",
                            action="store_true", default=False)
        parser.add_argument('-ufc', '--unlimited-fmg-cycles',
                            help="set max_v_cycles during f-cycles to 20 (that's nearly without limit :-)",
                            action="store_true", default=False)
        parser.add_argument('-dg', '--dump-grids', help='dump various grids for comparison with hpgmg.c',
                            action="store_true", default=False)
        parser.add_argument('-l', '--log', help='turn on logging', action="store_true", default=False)
        parser.add_argument('-b', '--backend', help='turn on JIT', choices=('python', 'c', 'omp', 'ocl'), default='python')
        parser.add_argument('-v', '--verbose', help='print verbose', action="store_true", default=False)
        parser.add_argument('-bd', '--blocking_dimensions', help='number of dimensions to block in', default=0, type=int)
        parser.add_argument('-bls', '--block_size', help='size of each block', default=32, type=int)
        parser.add_argument('-t', '--tune', help='try tuning it', default=False, action="store_true")
        finite_volume.CONFIG = parser.parse_args(args=args)
        finite_volume.CONFIG.block_hierarchy = (finite_volume.CONFIG.block_size,) * int(
            finite_volume.CONFIG.blocking_dimensions or 2
        )
        if finite_volume.CONFIG.backend == 'c':
            finite_volume.compiler = CCompiler()
        elif finite_volume.CONFIG.backend == 'omp':
            finite_volume.compiler = OpenMPCompiler()
        return finite_volume.CONFIG

    @staticmethod
    def get_solver(args=None):
        """
        this is meant for use in testing
        :return:
        """
        config = SimpleMultigridSolver.get_configuration(args=args)
        return SimpleMultigridSolver(config)

    @time_this
    @profile
    def benchmark_hpgmg(self, start_level=0):
        if self.backend == 'python':
            min_solves = 1
            number_passes = 1
        else:
            min_solves = 10
            number_passes = 1

        for pass_num in range(number_passes):
            if pass_num == 0:
                if self.backend == 'python':
                    print("===== Running python, no warm-up, one pass ============================".format(min_solves))
                else:
                    print("===== Warming up by running {:d} solves ===============================".format(min_solves))
            else:
                print("===== Running {:d} solves =============================================".format(min_solves))

            for solve_pass in range(min_solves):
                self.all_levels[start_level].fill_mesh(self.all_levels[start_level].cell_values, 0.0)

                if self.do_f_cycle:
                    self.solve_with_f_cycle(0, 0.0, 1.0e-10)
                else:
                    self.solve(start_level=start_level)
            if pass_num == 0 and self.backend != 'python':
                time_this.reset()

        print("===== Timing Breakdown ==============================================")
        self.show_timing_information()
        self.show_error_information()
        if self.compute_richardson_error:
            self.run_richardson_test()
        if self.configuration.verbose:
            print('Backend: {}'.format(self.configuration.backend))

    @staticmethod
    @time_this
    @profile
    def main():
        configuration = SimpleMultigridSolver.get_configuration()
        solver = SimpleMultigridSolver(configuration)
        solver.backend = configuration.backend

        # solver.solve()
        solver.benchmark_hpgmg()
        solver.show_timing_information()
        solver.show_error_information()
        if solver.compute_richardson_error:
            solver.run_richardson_test()
        if configuration.verbose:
            print('Backend: {}'.format(solver.configuration.backend))

if __name__ == '__main__':
    SimpleMultigridSolver.main()