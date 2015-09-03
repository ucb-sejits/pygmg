from __future__ import print_function
import sympy
import logging
from hpgmg.finite_volume.operators.specializers.util import profile, time_this

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Initializer(object):
    """
    Base class for PyGMG problem initializers, takes a level and fills it in
    """
    def __init__(self, solver):
        self.solver = solver
        self.index_to_cell_centered = solver.fine_level.coord_to_cell_center_point
        self.index_to_face_centered = solver.fine_level.coord_to_face_center_point

    def init_kernel(self):
        pass

    @time_this
    @profile
    def initialize_level_constant_coefficient(self, level):
        """
        Initialize the right_hand_side(VECTOR_F), exact_solution(VECTOR_UTRUE)
        alpha, and possibly the beta_faces
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0

        solver = self.solver
        problem = solver.problem
        beta_generator = solver.beta_generator

        level.alpha.fill(alpha)
        for dim in range(solver.dimensions):
            level.beta_face_values[dim].fill(beta)
        #fill U_TRUE, exact_solution
        solver.initialize_mesh(level, level.exact_solution, problem.expression, level.coord_to_cell_center_point)

        f_exp = self.a * alpha * problem.expression
        solver.initialize_mesh(level, level.exact_solution, f_exp, level.coord_to_cell_center_point)

        if level.alpha_is_zero is None:
            level.alpha_is_zero = level.dot_mesh(level.alpha, level.alpha) == 0.0
        logging.debug("level.alpha_is_zero {}".format(level.alpha_is_zero))

    @time_this
    @profile
    def initialize_level_variable_coefficient(self, level):
        """
        Initialize the right_hand_side(VECTOR_F), exact_solution(VECTOR_UTRUE)
        alpha, and possibly the beta_faces
        :param level:
        :return:
        """
        alpha = 1.0
        beta = 1.0

        solver = self.solver
        problem = solver.problem

        level.alpha.fill(alpha)
        for dim in range(solver.dimensions):
            level.beta_face_values[dim].fill(beta)
        #fill U
        solver.initialize_mesh(level, level.exact_solution, problem.expression, level.coord_to_cell_center_point)

        f_exp = self.a * alpha * problem.expression
        solver.initialize_mesh(level, level.exact_solution, f_exp, level.coord_to_cell_center_point)

        if level.alpha_is_zero is None:
            level.alpha_is_zero = level.dot_mesh(level.alpha, level.alpha) == 0.0
        logging.debug("level.alpha_is_zero {}".format(level.alpha_is_zero))

