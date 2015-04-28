from __future__ import print_function
from stencil_code.neighborhood import Neighborhood
from hpgmg.finite_volume.simple_level import SimpleLevel
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Restriction(object):
    RESTRICT_CELL = 0
    RESTRICT_FACE_I = 1
    RESTRICT_FACE_J = 2
    RESTRICT_FACE_K = 3

    def __init__(self, solver):
        self.solver = solver
        self.dimensions = self.solver.dimensions

        # self.neighbor_offsets = [
        #     [
        #         Coord(0, 0, 0),
        #         Coord(0, 0, 1),
        #         Coord(0, 1, 0),
        #         Coord(0, 1, 1),
        #         Coord(1, 0, 0),
        #         Coord(1, 0, 1),
        #         Coord(1, 1, 0),
        #         Coord(1, 1, 1),
        #     ],
        #     [Coord(0, 0, 0), Coord(0, 0, 1), Coord(0, 1, 0), Coord(0, 1, 1), ],
        #     [Coord(0, 0, 0), Coord(0, 0, 1), Coord(1, 0, 0), Coord(1, 0, 1), ],
        #     [Coord(0, 0, 0), Coord(0, 1, 0), Coord(1, 0, 0), Coord(1, 1, 0), ],
        # ]

        def all_positive(coord):
            return all(x >= 0 for x in coord)

        moore_neighborhood = Neighborhood.moore_neighborhood(radius=1, dim=self.dimensions, include_origin=True)
        self.neighbor_offsets = [
            filter(
                all_positive,
                moore_neighborhood,
            )
        ]
        for dim in range(self.dimensions):
            def all_positive_locked(coord):
                return coord[dim] == 0 and all(x >= 0 for x in coord)

            self.neighbor_offsets.append(
                filter(
                    all_positive_locked,
                    moore_neighborhood
                )
            )

    def restrict(self, level, target, source, restriction_type):
        assert(isinstance(level, SimpleLevel))
        for target_point in level.interior_points():
            source_point = (target_point * 2) - level.ghost_zone
            assert isinstance(target_point, Coord)
            assert isinstance(source_point, Coord)
            target[target_point] = 0.0
            for neighbor_offset in self.neighbor_offsets[restriction_type]:
                target[target_point] += source[source_point + neighbor_offset]
            target[target_point] *= (1.0 / len(self.neighbor_offsets[restriction_type]))
