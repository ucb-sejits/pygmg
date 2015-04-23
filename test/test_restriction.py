from __future__ import print_function
from hpgmg.finite_volume.operators.restriction import Restriction
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest


class TestRestriction(unittest.TestCase):
    def test_basics(self):
        solver = SimpleMultigridSolver.get_solver(["3"])

        restriction = Restriction(solver)

        neighbor_offsets_3d = [
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

        hand_coded = set(neighbor_offsets_3d[0])
        generated = set(restriction.neighbor_offsets[0])

        print("hand coded {}".format(hand_coded))
        print("generated  {}".format(generated))
        self.assertTrue(hand_coded == generated)

        for dim in range(3):
            hand_coded = set(neighbor_offsets_3d[dim+1])
            generated = set(restriction.neighbor_offsets[dim+1])

            print("hand coded {}".format(hand_coded))
            print("generated  {}".format(generated))
            self.assertTrue(hand_coded == generated)

