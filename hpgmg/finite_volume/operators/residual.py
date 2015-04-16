from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Residual(object):
    def __init__(self, solver):
        self.solver = solver
        self.a = self.solver.a
        self.b = self.solver.b
        self.operator = self.solver.problem_operator

    def run(self, level, target_mesh, source_mesh, right_hand_side):
        for index in level.interior_points():
            target_mesh[index] = right_hand_side[index] - self.operator.apply_op(source_mesh, index)