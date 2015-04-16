"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function

__author__ = 'nzhang-dev'

from hpgmg.finite_volume.space import Space, Vector
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.problem_sine import SineProblem


class SimpleLevel(object):
    """
    From hpgmg defines.h, the list of matrices at each level
    #define  VECTOR_TEMP         0 //
    #define  VECTOR_UTRUE        1 // exact solution used to generate f
    #define  VECTOR_F_MINUS_AV   2 // cell centered residual (f-Av)
    //------------------------------------------------------------------------------------------------------------------------------
    #define  VECTOR_F            3 // original right-hand side (Au=f), cell centered
    #define  VECTOR_U            4 // numerical solution
    #define  VECTOR_ALPHA        5 // cell centered coefficient
    #define  VECTOR_BETA_I       6 // face centered coefficient (n.b. element 0 is the left face of the ghost zone element)
    #define  VECTOR_BETA_J       7 // face centered coefficient (n.b. element 0 is the back face of the ghost zone element)
    #define  VECTOR_BETA_K       8 // face centered coefficient (n.b. element 0 is the bottom face of the ghost zone element)
    //------------------------------------------------------------------------------------------------------------------
    #define  VECTOR_DINV         9 // cell centered relaxation parameter (e.g. inverse of the diagonal)
    #define  VECTOR_L1INV       10 // cell centered relaxation parameter (e.g. inverse of the L1 norm of each row)
    #define  VECTOR_VALID       11 // cell centered array noting which cells are actually present

    """
    FACE_I = 0
    FACE_J = 0
    FACE_K = 0

    def __init__(self, solver, space, level_number=0):
        self.solver = solver
        self.space = space + (solver.ghost_zone * 2)

        self.configuration = solver.configuration
        self.is_variable_coefficient = not solver.configuration.fixed_beta
        self.problem_name = solver.configuration.problem
        self.level_number = level_number
        if self.problem_name == 'sine':
            self.problem = SineProblem
        self.ghost_zone = solver.ghost_zone

        self.cell_values = Mesh(self.space)
        self.right_hand_side = Mesh(self.space)
        self.exact_solution = Mesh(self.space) if self.level_number == 0 else None

        self.alpha = Mesh(self.space)
        self.beta_face_values = [
            Mesh(self.space),
            Mesh(self.space),
            Mesh(self.space),
        ]
        self.valid = Mesh(self.space)
        for index in self.interior_points():
            self.valid[index] = 1.0

        self.d_inverse = Mesh(self.space)
        self.l1_inverse = Mesh(self.space)
        self.temp = Mesh(self.space)
        self.residual = Mesh(self.space)

        self.dominant_eigen_value_of_d_inv_a = 0.0


        self.cell_size = 1.0 / self.space[0]

    def make_coarser_level(self):
        coarser_level = SimpleLevel(self.solver, (self.space-self.ghost_zone)//2, self.level_number+1)
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

        for element_index in self.interior_points():
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

            self.right_hand_side[element_index] = f
            self.exact_solution[element_index] = u

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


