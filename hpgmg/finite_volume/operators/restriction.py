from __future__ import print_function
from hpgmg.finite_volume.space import Vector

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Restriction(object):
    RESTRICT_CELL = 0
    RESTRICT_FACE_I = 1
    RESTRICT_FACE_J = 2
    RESTRICT_FACE_K = 3

    def __init__(self):
        self.neighbor_offsets = [
            [
                Vector(0, 0, 0),
                Vector(0, 0, 1),
                Vector(0, 1, 0),
                Vector(0, 1, 1),
                Vector(1, 0, 0),
                Vector(1, 0, 1),
                Vector(1, 1, 0),
                Vector(1, 1, 1),
            ],
            [Vector(0, 0, 0), Vector(0, 0, 1), Vector(0, 1, 0), Vector(0, 1, 1), ],
            [Vector(0, 0, 0), Vector(0, 0, 1), Vector(1, 0, 0), Vector(1, 0, 1), ],
            [Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 0, 0), Vector(1, 1, 0), ],
        ]

    def restrict(self, target, source, restriction_type):
        for target_point in target.space.points:
            source_point = target_point * 2
            target[target_point] = 0.0
            for neighbor_offset in self.neighbor_offsets[restriction_type]:
                target[target_point] += source[source_point + neighbor_offset]
            target[target_point] *= (1.0 / len(self.neighbor_offsets[restriction_type]))
