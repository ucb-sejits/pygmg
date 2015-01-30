from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.space import Space,  Coord


class TestSpace(unittest.TestCase):
    def test_aaa_create(self):
        space = Coord(0, 0, 0)
        self.assertEqual((space.i, space.j, space.k), (0, 0, 0))

        space = Space(1, 2, 3)
        self.assertEqual((space.i, space.j, space.k), (1, 2, 3))

    def test_indexing(self):
        sizes = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 3, 4),
            (4, 3, 2),
            (23, 52, 71),
        ]
        for a, b, c in sizes:
            space = Space(a, b, c)
            self.assertEqual(len(list(space.foreach())), space.volume())
            for index in space.foreach():
                index_1d = space.index_3d_to_1d(index)
                index_3d = Space(space.index_1d_to_3d(index_1d))
                self.assertEqual(index, index_3d)

    def test_multiply(self):
        self.assertEqual(Space(1, 1, 1) * 4, Space(4, 4, 4))
        self.assertEqual(Space(3, 2, 1) * 4, Space(12, 8, 4))
        self.assertEqual(Space(1, 2, 3) * 8, Space(8, 16, 24))
        self.assertEqual(4 * Space(1, 1, 1), Space(4, 4, 4))
        self.assertEqual(5 * Space(3, 2, 1), Space(15, 10, 5))
        self.assertEqual(1 * Space(1, 2, 3), Space(1, 2, 3))

    def test_add(self):
        self.assertEqual(Coord(1, 1, 1) + Coord(1, 2, 3), Coord(2, 3, 4))
        self.assertEqual(Coord(1, 1, 1) + (1, 2, 3), Coord(2, 3, 4))
        self.assertEqual(Coord(1, 1, 1) + 4, Coord(5, 5, 5))
        self.assertEqual(4 + Coord(1, 1, 1), Coord(5, 5, 5))

    def test_floor_div(self):
        self.assertEqual(Coord(12, 10, 9) // Coord(4, 5, 3), Coord(3, 2, 3))
        self.assertEqual(Coord(8, 6, 5) // (4, 2, 2), Coord(2, 3, 2))
        self.assertEqual(Coord(7, 8, 9) // 4, Coord(1, 2, 2))

    def test_neighbors(self):
        coord = Coord(1, 1, 1)
        self.assertEqual(len(list(coord.legal_neighbor_coords())), 26)
        origin = Coord(0, 0, 0)
        self.assertEqual(len(list(origin.legal_face_neighbor_coords())), 3)
        self.assertEqual(len(list(origin.legal_edge_neighbor_coords())), 3)
        self.assertEqual(len(list(origin.legal_corner_neighbor_coords())), 1)
        self.assertEqual(len(list(origin.legal_neighbor_coords())), 7)