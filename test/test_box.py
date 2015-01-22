from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.box import Box


class TestBox(unittest.TestCase):
    def test_create(self):
        box = Box(1, 100, 1, 1)
        self.assertTrue(box is not None)
        self.assertEqual(box.j_stride, 3)
        self.assertEqual(box.i_stride, 9)

        box.vectors[50] = 3.14

        box.add_vectors(10)
        self.assertEqual(box.vectors.size, 110)
        self.assertEqual(box.vectors[50], 3.14)