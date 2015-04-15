"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function

__author__ = 'nzhang-dev'

from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.problem_sine import SineProblem


class SimpleLevel(object):
    FACE_I = 0
    FACE_J = 0
    FACE_K = 0

    def __init__(self, solver, space, level_number=0):
        self.solver = solver
        self.space = space
        self.configuration = solver.configuration
        self.is_variable_coefficient = not solver.configuration.fixed_beta
        self.problem_name = solver.configuration.problem
        self.level_number = level_number
        if self.problem_name == 'sine':
            self.problem = SineProblem
        self.ghost_zone = solver.ghost_zone

        self.cell_values = Mesh(space)
        self.alpha = Mesh(space)
        self.beta_face_values = [
            Mesh(space),
            Mesh(space),
            Mesh(space),
        ]
        self.valid = Mesh(space)
        for index in self.interior_points():
            self.valid[index] = 1.0

        self.d_inverse = Mesh(space)
        self.l1_inverse = Mesh(space)
        self.temp = Mesh(space)
        self.dominant_eigen_value_of_d_inv_a = 0.0

        if self.level_number == 0:
            self.true_solution = Mesh(space)

        self.cell_size = 1.0 / space[0]

    def make_coarser_level(self):
        coarser_level = SimpleLevel(self.solver, self.space//2, self.level_number+1)
        return coarser_level

    @property
    def h(self):
        return self.cell_size

    def interior_points(self):
        for point in self.space.interior_points(self.ghost_zone):
            yield point

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


