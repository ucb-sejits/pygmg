from __future__ import print_function
import math
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.solvers.iterative_solver import IterativeSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'
"""
from
-----------------------------------------------------------------------------------------------
 Samuel Williams
 SWWilliams@lbl.gov
 Lawrence Berkeley National Lab
-----------------------------------------------------------------------------------------------
"""


class BiCGStab(IterativeSolver):
    def __init__(self, solver, desired_reduction):
        self.solver = solver
        self.krylov_diagonal_precondition = True
        self.desired_reduction_in_norm = desired_reduction

    def solve(self, level, target_mesh, residual_mesh):
        """
        Algorithm 7.7 in Iterative Methods for Sparse Linear Systems(Yousef Saad)
        Algorithm 1 in Analysis and Practical use of Flexible BiCGStab (Jie Chen)

        :param target_mesh:
        :param residual_mesh:
        :return:
        """

        shape = target_mesh.space
        r0_mesh, r_mesh, p_mesh = Mesh(shape), Mesh(shape), Mesh(shape)
        q_mesh, s_mesh, t_mesh = Mesh(shape), Mesh(shape), Mesh(shape)
        ap_mesh, as_mesh = Mesh(shape), Mesh(shape)

        j_max, j = 200, 0
        bicgstab_failed, bicgstab_converged = 0, False

        top_solver = self.solver

        # r0[] = R_id[] - A(x_id)
        # r[] = r0[]
        # p[] = r0[]
        # r_dot_r0 = dot(r, r0)
        # norm of initial residual

        top_solver.residual.run(level, r0_mesh, target_mesh, residual_mesh)
        level.scale_mesh(r_mesh, 1.0, r0_mesh)
        level.scale_mesh(p_mesh, 1.0, r0_mesh)
        r_dot_r0 = level.dot_mesh(r_mesh, r0_mesh)
        norm_of_r0 = level.norm_mesh(r_mesh)
        bicgstab_converged = r_dot_r0 == 0.0 or norm_of_r0 == 0.0

        while j < j_max and not bicgstab_failed and not bicgstab_converged:
            j += 1
            level.krylov_iterations += 1
            if self.krylov_diagonal_precondition:
                #  q[] = Dinv[]*p[]
                level.multiply_meshes(q_mesh, 1.0, level.d_inverse, p_mesh)
            else:
                # q[] = p[]
                level.scale_mesh(q_mesh, 1.0, p_mesh)

            #Ap[] = AM^{-1}(p)
            top_solver.boundary_updater.apply(level, q_mesh)
            for index in level.interior_points():
                ap_mesh[index] = top_solver.problem_operator.apply_op(q_mesh, index, level)

            # Ap_dot_r0 = dot(Ap,r0)
            # pivot breakdown ???
            # alpha = r_dot_r0 / Ap_dot_r0
            # pivot breakdown ???
            # x_id[] = x_id[] + alpha*q[]
            # s[]    = r[]    - alpha*Ap[]   (intermediate residual?)
            # FIX - redundant??  norm of intermediate residual
            # FIX - redundant??  if As_dot_As==0, then As must be 0 which implies s==0

            ap_dot_r0 = level.dot_mesh(ap_mesh, r0_mesh)
            if ap_dot_r0 == 0.0:
                bicgstab_failed = 1
                break
            alpha = r_dot_r0 / ap_dot_r0
            if math.isinf(alpha):
                bicgstab_failed = 2
                break

            level.add_meshes(target_mesh, 1.0, target_mesh, alpha, q_mesh)
            level.add_meshes(s_mesh, 1.0, r_mesh, -alpha, ap_mesh)

            norm_of_s = level.norm_mesh(s_mesh)
            if norm_of_s == 0.0 or norm_of_r0 < self.desired_reduction_in_norm * norm_of_r0:
                bicgstab_converged = True
                break

            if self.krylov_diagonal_precondition:
                # t[] = Dinv[] * s[]
                level.multiply_meshes(t_mesh, 1.0, level.d_inverse, s_mesh)
            else:
                # t[] = s[]
                level.scale_mesh(t_mesh, 1.0, s_mesh)

            # As = AM^{-1}(s)
            # As_dot_As = dot(As,As)
            # As_dot_s  = dot(As, s)
            # converged ?
            # omega = As_dot_s / As_dot_As
            # stabilization breakdown ???
            # stabilization breakdown ???
            # x_id[] = x_id[] + omega*t[]
            # r[]    = s[]    - omega*As[]  (recursively computed / updated residual)
            # norm of recursively computed residual (good enough??)

            top_solver.boundary_updater.apply(level, t_mesh)
            for index in level.interior_points():
                as_mesh[index] = top_solver.problem_operator.apply_op(t_mesh, index, level)
            as_dot_as = level.dot_mesh(as_mesh, as_mesh)
            as_dot_s = level.dot_mesh(as_mesh, s_mesh)
            if as_dot_as == 0.0:
                bicgstab_converged = True
                break
            omega = as_dot_s / as_dot_as
            if omega == 0.0:
                bicgstab_failed = 3
                break
            if math.isinf(omega):
                bicgstab_failed = 4
                break

            level.add_meshes(target_mesh, 1.0, target_mesh, omega, t_mesh)
            level.add_meshes(r_mesh, 1.0, s_mesh, -omega, as_mesh)
            norm_of_r = level.norm_mesh(r_mesh)
            if norm_of_r == 0.0 or norm_of_r < self.desired_reduction_in_norm * norm_of_r0:
                bicgstab_converged = True
                break

            if top_solver.configuration.log:
                #ifdef __DEBUG                                                              //
                top_solver.residual.run(level.temp, target_mesh, residual_mesh)
                norm_of_residual = level.norm_mesh(level.temp)
                print("j={:8d}, norm={:12.6e}, norm_initial={:12.6e}, reduction={:e}".format(
                    j, norm_of_residual, norm_of_r0, norm_of_residual/norm_of_r0
                ))

            # r_dot_r0_new = dot(r,r0)
            # Lanczos breakdown ???
            # beta = (r_dot_r0_new/r_dot_r0) * (alpha/omega)
            # ???
            # VECTOR_TEMP = (p[]-omega*Ap[])
            # p[] = r[] + beta*(p[]-omega*Ap[])
            # r_dot_r0 = r_dot_r0_new   (save old r_dot_r0)

            r_dot_r0_new = level.dot_mesh(r_mesh, r0_mesh)
            if r_dot_r0_new == 0.0:
                bicgstab_failed = 5
                break
            beta = r_dot_r0_new/r_dot_r0 * (alpha/omega)
            if math.isinf(beta):
                bicgstab_failed = 6
                break
            level.add_meshes(level.temp, 1.0, p_mesh, -omega, ap_mesh)
            level.add_meshes(p_mesh, 1.0, r_mesh, beta, level.temp)
            r_dot_r0 = r_dot_r0_new
        if bicgstab_failed > 0:
            print("BiCGStab Failed... error = {:d}".format(bicgstab_failed))