from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class CycleCounter(object):
    def __init__(self):
        self.smooth = 0.0
        self.apply_op = 0.0
        self.residual = 0.0
        self.blas1 = 0.0
        self.blas2 = 0.0
