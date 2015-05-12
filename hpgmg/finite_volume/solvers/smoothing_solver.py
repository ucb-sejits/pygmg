from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class SmoothingSolver(object):
    def __init__(self, solver, desired_reduction):
        self.solver = solver
        self.a = solver.a
        self.b = solver.b
        self.desired_reduction = desired_reduction

    def solve(self, level, target_mesh, residual_mesh):
        if level.alpha_is_zero is None:
            level.alpha_is_zero = level.dot_mesh(level.alpha, level.alpha)

        self.solver.residual.run(level, level.temp, target_mesh, residual_mesh)
        level.multiply_meshes(level.temp, scale_factor=1.0, mesh_a=level.temp, mesh_b=level.d_inverse)

        norm_of_r0 = level.norm_mesh(level.temp)

        smooth_count, max_smooths, converged = 0, 10, False
        while smooth_count < max_smooths and not converged:
            smooth_count += 1
            self.solver.smoother.smooth(level, target_mesh, residual_mesh)
            self.solver.residual.run(level, level.temp, target_mesh, residual_mesh)
            level.multiply_meshes(level.temp, scale_factor=1.0, mesh_a=level.temp, mesh_b=level.d_inverse)
            norm_of_r = level.norm_mesh(level.temp)
            if norm_of_r == 0.0 or norm_of_r < self.desired_reduction * norm_of_r0:
                converged = True
