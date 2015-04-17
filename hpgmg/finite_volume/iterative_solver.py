from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class IterativeSolver(object):
    def __init__(self, solver, desired_reduction):
        self.solver = solver
        self.a = solver.a
        self.b = solver.b
        self.desired_reduction = desired_reduction

    def solve(self, level, target_mesh, source_mesh):
        if level.alpha_is_zero is None:
            level.alpha_is_zero = dot(level, level.alpha, level.alpha)

        self.solver.residual.run(level, level.temp, target_mesh, source_mesh)
        self.solver.multiply_vectors()
