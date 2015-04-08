"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function
import argparse
import os
import cProfile
from hpgmg.finite_volume.operators.restriction import Restriction

__author__ = 'nzhang-dev'

import numpy as np

from hpgmg.finite_volume.space import Coord, Space, Vector
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.operators.problem_sine import SineProblem
from hpgmg.finite_volume.smoothers import gauss_siedel


def simple_restrict(mesh):
    slices = tuple(slice(None, None, 2) for _ in range(mesh.ndim))
    return mesh[slices]


class SimpleLevel(object):
    FACE_I = 0
    FACE_J = 0
    FACE_K = 0

    def __init__(self, space, level_number=0, configuration=None):
        self.space = space
        self.configuration = configuration
        self.is_variable_coefficient = not configuration.fixed_beta
        self.problem_name = configuration.problem
        self.level_number = level_number
        if self.problem_name == 'sine':
            self.problem = SineProblem

        self.cell_values = Mesh(space)
        self.alpha = Mesh(space)
        self.beta_face_values = [
            Mesh(space),
            Mesh(space),
            Mesh(space),
        ]
        if self.level_number == 0:
            self.true_solution = Mesh(space)

        self.cell_size = 1.0 / space[0]

    def make_coarser_level(self):
        coarser_level = SimpleLevel(self.space//2, self.level_number+1, self.configuration)
        return coarser_level

    @property
    def h(self):
        return self.cell_size

    def initialize(self, a=1.0, b=1.0):
        alpha = 1.0
        beta = 1.0
        beta_xyz = Vector(0.0, 0.0, 0.0)
        beta_i, beta_j, beta_k = 1.0, 1.0, 1.0

        problem = self.problem

        for element_index in self.space.points:
            coord = half_cell = Vector([0.5 for _ in self.space])
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
            self.true_solution[element_index] = a * alpha * u
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


class SimpleMultigridSolver(object):
    def __init__(self, interpolate_function, restrict_function,
                 bottom_solver_function, smooth_function, smooth_iterations):
        self.interpolate_function = interpolate_function
        self.restrict_function = restrict_function
        self.bottom_solver_function = bottom_solver_function
        self.smooth_function = smooth_function
        self.smooth_iterations = smooth_iterations

    def v_cycle(self, level):
        if min(level.space) < 3:
            self.bottom_solver_function(level)
            return

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
            self.smooth_function(new_cell_values, coarser_level.cell_values)
            coarser_level.cell_values = new_cell_values

        self.v_cycle(coarser_level)

        self.interpolate_function(coarser_level.cell_values, level.cell_values, Restriction.RESTRICT_CELL)
        self.interpolate_function(coarser_level.beta_face_values[SimpleLevel.FACE_I],
                               level.beta_face_values[SimpleLevel.FACE_I], Restriction.RESTRICT_FACE_I)
        self.interpolate_function(coarser_level.beta_face_values[SimpleLevel.FACE_J],
                               level.beta_face_values[SimpleLevel.FACE_J], Restriction.RESTRICT_FACE_J)
        self.interpolate_function(coarser_level.beta_face_values[SimpleLevel.FACE_K],
                               level.beta_face_values[SimpleLevel.FACE_K], Restriction.RESTRICT_FACE_K)


    def MGV(self, A, b, x):
        """
        eigen_vectors cycle

        :param A: A in Ax=b
        :param b: b in Ax=b
        :param x: guess for x
        :return: refined guess for X in Ax = b
        """
        if min(b.shape) <= 3: # minimum size, so we compute exact
            return np.linalg.solve(A, b.flatten()).reshape(x.shape)

        x = self.smooth(A, b, x, self.smooth_iterations)
        residual = (np.dot(A, x.flatten()) - b.flatten()).reshape(x.shape)
        restricted_residual = self.restrict(residual)
        diff = self.interpolate(self(self._evolve(A, b.ndim), restricted_residual, np.zeros_like(restricted_residual)))
        x -= diff
        result = self.smooth(A, b, x, self.smooth_iterations)
        return result    

    def FMG(self, A, b, x=None):
        """
        Perform full multi-grid
        :param A: A in Ax=b
        :param b: b in Ax=b
        :param x: guess for x
        :return: refined guess for X in Ax = b
        """
        matrices = []
        Amatrices = []
        b_restrict = b[:]
        A_restrict = A[:]
        matrices.append(b_restrict)
        Amatrices.append(A_restrict)

        #continuously restrict b, and store all intermediate b's in list
        while(not (min(b_restrict.shape) <= 3)):
            b_restrict = self.restrict(b_restrict)
            A_restrict = self._evolve(A_restrict, b.ndim)
            matrices.append(b_restrict)
            Amatrices.append(A_restrict)

        #base case
        shape = b_restrict.shape
        x = np.linalg.solve(A_restrict, b_restrict.flatten())
        x = x.reshape(shape)

        for i in range(len(matrices)-2,-1,-1):
            x = self.MGV(Amatrices[i], matrices[i], interpolate(x))
        return x

    def __call__(self, level, cycle="eigen_vectors"):
        if cycle == "eigen_vectors":
            return self.MGV(A, b, x)  # FIXME
        elif cycle == "F":
            return self.FMG(A, b, x)  # FIXME

    @staticmethod
    def _evolve(A, ndim):
        n = int(round((A.shape[0])**(1/ndim)))
        n = int((n + 1)/2)
        slices = tuple(slice(0, n**ndim) for _ in range(A.ndim))
        return A[slices]

    def profiled_call(self, A, b, x):
        cProfile.runctx('self(A, b, x)', {'self': self, 'A':A, 'b':b, 'x':x}, {})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log2_level_size', help='Size of space will be 3d, each dim will be of 2^(log2_level_size)',
                        default=6, type=int)
    parser.add_argument('-p', '--problem',
                        help="problem name, one of [sine]",
                        default='sine',
                        choices=['sine'], )
    parser.add_argument('-bc', '--boundary-conditions', dest='boundary_condition',
                        help="Type of boundary condition. Use p for Periodic and d for Dirichlet. Default is d",
                        default=('p' if os.environ.get('USE_PERIODIC_BC', 0) else 'd'),
                        choices=['p', 'd'], )
    parser.add_argument('-fb', '--fixed_beta', dest='fixed_beta', action='store_true',
                        help="Use 1.0 as fixed value of beta, default is variable beta coefficient",
                        default=False, )
    command_line_configuration = parser.parse_args()

    global_size = Space([2**command_line_configuration.log2_level_size for _ in range(3)])

    restrictor = Restriction().restrict

    fine_level = SimpleLevel(global_size, 0, command_line_configuration)
    fine_level.initialize()
    fine_level.print()

    solver = SimpleMultigridSolver(interpolate_m, restrictor, None, gauss_siedel, 4)
    print(solver.v_cycle(fine_level))
