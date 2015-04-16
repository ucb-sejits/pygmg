from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.space import Space, Coord, Vector


class TestSpace(unittest.TestCase):
    def test_aaa_create(self):
        space = Coord(0, 0, 0)
        self.assertEqual(space, (0, 0, 0))

        space = Space(1, 2, 3)
        self.assertEqual(space, (1, 2, 3))

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
            self.assertEqual(len(list(space.points)), space.volume)
            for index in space.points:
                index_1d = space.index_to_1d(index)
                index_3d = Space(space.index_from_1d(index_1d))
                self.assertEqual(index, index_3d)

    def test_multiply(self):
        self.assertEqual(Space(2, 2, 2) * 4, Space(8, 8, 8))

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
        space = Space(4, 3, 4)
        self.assertEqual(len(list(space.neighbors(coord))), 27)
        self.assertEqual(len(list(space.neighbors(coord, 0))), 1)
        self.assertEqual(len(list(space.neighbors(coord, 1))), 6)
        self.assertEqual(len(list(space.neighbors(coord, 2))), 12)
        self.assertEqual(len(list(space.neighbors(coord, 3))), 8)

    def test_strides(self):
        self.assertEqual(Space([5]).strides(), [1])
        self.assertEqual(Space([7, 8]).strides(), [8, 1])
        self.assertEqual(Space([99, 6, 5]).strides(), [30, 5, 1])

    def test_coord_vs_vector(self):
        self.assertEqual(Vector(1, 1, 1) / 2, Vector(.5, .5 ,.5), "vectors support real numbers")

        self.assertEqual(Coord(1, 1, 1) / 2, Coord(0, 0, 0), "coords are only ints")
        self.assertEqual(Coord(.5, .5, .5), Coord(0, 0, 0), "coords are silently converted to ints")

        self.assertEqual(Vector(1, 2, 3) * Vector(3, 4, 5), 26, "vector vector multiply is dot")
        self.assertEqual(Coord(1, 2, 3) * Coord(3, 4, 5), Coord(3, 8, 15), "coord coord multiply is element-wise")

    def test_interior_points(self):
        space = Space(4, 4)
        ghost = Space(1, 1)

        for point in space.interior_points(ghost):
            print(point)