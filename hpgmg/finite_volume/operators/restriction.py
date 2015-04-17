from __future__ import print_function
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Restriction(object):
    RESTRICT_CELL = 0
    RESTRICT_FACE_I = 1
    RESTRICT_FACE_J = 2
    RESTRICT_FACE_K = 3

    def __init__(self):
        self.neighbor_offsets = [
            [
                Coord(0, 0, 0),
                Coord(0, 0, 1),
                Coord(0, 1, 0),
                Coord(0, 1, 1),
                Coord(1, 0, 0),
                Coord(1, 0, 1),
                Coord(1, 1, 0),
                Coord(1, 1, 1),
            ],
            [Coord(0, 0, 0), Coord(0, 0, 1), Coord(0, 1, 0), Coord(0, 1, 1), ],
            [Coord(0, 0, 0), Coord(0, 0, 1), Coord(1, 0, 0), Coord(1, 0, 1), ],
            [Coord(0, 0, 0), Coord(0, 1, 0), Coord(1, 0, 0), Coord(1, 1, 0), ],
        ]

    def restrict(self, level, target, source, restriction_type):
        assert(isinstance(level, SimpleLevel))
        for target_point in level.interior_points():
            source_point = target_point * 2
            assert isinstance(target_point, Coord)
            assert isinstance(source_point, Coord)
            target[target_point] = 0.0
            for neighbor_offset in self.neighbor_offsets[restriction_type]:
                target[target_point] += source[source_point + neighbor_offset]
            target[target_point] *= (1.0 / len(self.neighbor_offsets[restriction_type]))
