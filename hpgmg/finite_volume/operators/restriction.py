from __future__ import print_function

from stencil_code.neighborhood import Neighborhood
from hpgmg.finite_volume.operators.specializers.restrict_specializer import CRestrictSpecializer
from hpgmg.finite_volume.operators.specializers.util import profile, time_this, specialized_func_dispatcher

from hpgmg.finite_volume.simple_level import SimpleLevel


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Restriction(object):
    RESTRICT_CELL = 0
    RESTRICT_FACE_I = 1
    RESTRICT_FACE_J = 2
    RESTRICT_FACE_K = 3

    def __init__(self, solver):
        self.solver = solver
        self.dimensions = self.solver.dimensions
        self.__kernels = {}

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
                return coord[dim] == 0 and all_positive(coord)

            self.neighbor_offsets.append(
                filter(
                    all_positive_locked,
                    moore_neighborhood
                )
            )
        self.neighbor_offsets = tuple(tuple(i) for i in self.neighbor_offsets)

    @time_this
    @specialized_func_dispatcher({
        'c': CRestrictSpecializer,
        'omp': CRestrictSpecializer
    })
    def restrict_std(self, level, target, source, restriction_type):
        #assert(isinstance(level, SimpleLevel))

        if restriction_type == self.RESTRICT_CELL:
            #print("CELL:", level.interior_points())
            for target_point in level.interior_points():
                source_point = (target_point * 2) - level.ghost_zone
                # target[target_point] = 0.0
                # for neighbor_offset in self.neighbor_offsets[restriction_type]:
                #     target[target_point] += source[source_point + neighbor_offset]
                # target[target_point] *= (1.0 / len(self.neighbor_offsets[restriction_type]))

                target[target_point] = sum(
                    source[source_point + neighbor_offset] for neighbor_offset in self.neighbor_offsets[restriction_type]
                ) / len(self.neighbor_offsets[restriction_type])
        else:
            #print(restriction_type, level.beta_interpolation_points(restriction_type-1))
            for target_point in level.beta_interpolation_points(restriction_type-1):
                #print(target_point)
                source_point = (target_point * 2) - level.ghost_zone
                # target[target_point] = 0.0
                # for neighbor_offset in self.neighbor_offsets[restriction_type]:
                #     target[target_point] += source[source_point + neighbor_offset]
                # target[target_point] *= (1.0 / len(self.neighbor_offsets[restriction_type]))
                target[target_point] = sum(
                    source[source_point + neighbor_offset] for neighbor_offset in self.neighbor_offsets[restriction_type]
                ) / len(self.neighbor_offsets[restriction_type])


    restrict = restrict_std