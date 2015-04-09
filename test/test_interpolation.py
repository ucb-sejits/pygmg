from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.operators.interpolation import InterpolatorPQ
from hpgmg.finite_volume.space import Space, Coord, Vector


class TestInterpolationPQ(unittest.TestCase):
    def test_interpolation_pq_neighbor_stuff(self):
        interpolator = InterpolatorPQ()

        self.assertEqual(len(interpolator.convolution), 27, "this interpolator uses full 3d moore neighborhood")
        self.assertEqual(interpolator.convolution[0][0], -27.0, "coefficient of first neighbor is -27.0")
        self.assertEqual(interpolator.convolution[0][1][0], Coord(-1, -1, -1),
                         "neighbor offset for all evens in mesh index is")

    def test_backward_forward_neighbor_ideas(self):
        """
        when interpolating with pq method, depending on the eveness of each axis for a given point in mesh
        each neighbor offsets will have their value inverted along that axis, a table is created in the interpolator
        that assigns a category to the index from 0 to 7, a value for each possible combination of odd/even for a 3d
        coordinate
        :return:
        """
        all_even_index = Coord(0, 0, 0)
        self.assertEqual(InterpolatorPQ.compute_neighbor_index(all_even_index), 0, "all even index is category 0")
        k_odd = Coord(10, 1022, 7)
        self.assertEqual(InterpolatorPQ.compute_neighbor_index(k_odd), 1, "if just k is odd then category 1")
        all_odd = Coord(3, 5, 7)
        self.assertEqual(InterpolatorPQ.compute_neighbor_index(all_odd), 7, "if all are odd then category 7")

        interpolator = InterpolatorPQ()
        coefficient_0, offsets_0 = interpolator.convolution[0]

        all_even_offset = offsets_0[InterpolatorPQ.compute_neighbor_index(all_even_index)]
        k_odd_offset = offsets_0[InterpolatorPQ.compute_neighbor_index(k_odd)]

        self.assertEqual(all_even_offset.i, k_odd_offset.i)
        self.assertEqual(all_even_offset.j, k_odd_offset.j)
        self.assertNotEqual(all_even_offset.k, k_odd_offset.k)
