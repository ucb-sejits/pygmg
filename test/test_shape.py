from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.shape import Shape


class TestShape(unittest.TestCase):
    def test_create(self):
        shape = Shape()
        self.assertEqual((shape.i, shape.j, shape.k), (0, 0, 0))

        shape = Shape(1, 2, 3)
        self.assertEqual((shape.i, shape.j, shape.k), (1, 2, 3))

    def test_indexing(self):
        sizes = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 3, 4),
            (4, 3, 2),
            (23, 52, 71),
        ]
        for a, b, c in sizes:
            shape = Shape(a, b, c)
            self.assertEqual(len(list(shape.foreach())), shape.volume())
            for index in shape.foreach():
                index_1d = shape.index_3d_to_1d(index)
                index_3d = shape.index_1d_to_3d(index_1d)
                self.assertEqual(index, index_3d)

    def test_multiply(self):
        self.assertEqual(Shape(1, 1, 1) * 4, Shape(4, 4, 4))
        self.assertEqual(Shape(3, 2, 1) * 4, Shape(12, 8, 4))
        self.assertEqual(Shape(1, 2, 3) * 8, Shape(8, 16, 24))
        self.assertEqual(4 * Shape(1, 1, 1), Shape(4, 4, 4))
        self.assertEqual(5 * Shape(3, 2, 1), Shape(15, 10, 5))
        self.assertEqual(1 * Shape(1, 2, 3), Shape(1, 2, 3))
