"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg.finite_volume.space import Vector, Space
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.problem_sine import SineProblem


class SimpleLevel(object):
    """
    From hpgmg defines.h, the list of matrices at each level
    C vector offset    python mesh name
    ================== ====================================
    VECTOR_TEMP        temp
    VECTOR_UTRUE       exact_solution
    VECTOR_F_MINUS_AV  residual
    VECTOR_F           right_hand_side
    VECTOR_U           cell_values
    VECTOR_ALPHA       alpha
    VECTOR_BETA_I      beta_face_values[SimpleLevel.FACE_I]
    VECTOR_BETA_J      beta_face_values[SimpleLevel.FACE_J]
    VECTOR_BETA_K      beta_face_values[SimpleLevel.FACE_K]
    VECTOR_DINV        d_inverse
    VECTOR_L1INV       l1_inverse
    VECTOR_VALID       valid

    """
    FACE_I = 0
    FACE_J = 0
    FACE_K = 0

    def __init__(self, solver, space, level_number=0):
        assert(isinstance(space, Space))
        self.solver = solver
        self.space = space + (solver.ghost_zone * 2)

        self.configuration = solver.configuration
        self.is_variable_coefficient = not solver.configuration.fixed_beta
        self.problem_name = solver.configuration.problem
        self.level_number = level_number
        self.krylov_iterations = 0

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
        self.alpha_is_zero = None

    def make_coarser_level(self):
        coarser_level = SimpleLevel(self.solver, (self.space-self.ghost_zone)//2, self.level_number+1)
        return coarser_level

    @property
    def h(self):
        return self.cell_size

    def indices(self):
        for point in self.space.points:
            yield point

    def interior_points(self):
        for point in self.space.interior_points(self.ghost_zone):
            yield point

    def valid_indices(self):
        for index in self.interior_points():
            yield index

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

    def initialize_valid(self):
        for index in self.valid.indices():
            self.valid[index] = 1.0
            if self.solver.boundary_is_dirichlet and self.valid.space.is_boundary_point(index):
                self.valid[index] = 0.0

    def fill_mesh(self, mesh, value):
        for index in self.indices():
            mesh[index] = value if self.valid[index] > 0.0 else 0.0

    def add_meshes(self, target_mesh, scale_a, mesh_a, scale_b, mesh_b):
        for index in self.valid_indices():
            target_mesh[index] = scale_a * mesh_a[index] + scale_b + mesh_b[index]

    def multiply_meshes(self, target_mesh, scale_factor, mesh_a, mesh_b):
        for index in self.valid_indices():
            target_mesh[index] = scale_factor * mesh_a[index] * mesh_b[index]

    def invert_mesh(self, target_mesh, scale_factor, mesh_to_invert):
        for index in self.valid_indices():
            target_mesh[index] = scale_factor / mesh_to_invert

    def scale_mesh(self, target_mesh, scale_factor, source_mesh):
        for index in self.valid_indices():
            target_mesh[index] = scale_factor * source_mesh

    def shift_mesh(self, target_mesh, shift_value, source_mesh):
        for index in self.valid_indices():
            target_mesh[index] = shift_value * source_mesh

    def dot_mesh(self, mesh_a, mesh_b):
        accumulator = 0.0
        for index in self.valid_indices():
            accumulator += mesh_a[index] * mesh_b[index]
        return accumulator

    def norm_mesh(self, mesh):
        max_norm = 0.0
        for index in self.valid_indices():
            if abs(mesh[index]) > max_norm:
                max_norm = abs(mesh[index])
        return max_norm

    def project_cell_to_face(self, cell_mesh, face_id):
        lower_neighbor = Vector(
            -1 if d == face_id else 0 for d in range(len(self.space))
        )
        for index in self.valid_indices():
            self.beta_face_values[index] = 0.5 * (cell_mesh[index + lower_neighbor] + cell_mesh[index])

    def mean_mesh(self, mesh):
        """

        :param mesh:
        :return:
        """
        # TODO: original used computed cell count, is this over valid or what
        accumulator = 0.0
        cell_count = 0
        for index in self.valid_indices():
            accumulator += mesh[index]
            cell_count += 1
        return accumulator / cell_count

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
