from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Level(object):
    FACE_I = 0
    FACE_J = 0
    FACE_K = 0

    def __init__(self, space):
        self.cell_values = Mesh(space)
        self.beta_face_values = [
            Mesh(space),
            Mesh(space),
            Mesh(space),
        ]
        self.true_solution = Mesh(space)

        self.cell_size = 1.0 / space[0]

    def h(self):
        return self.cell_size