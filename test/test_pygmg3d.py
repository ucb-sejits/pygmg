from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.pymg3d import interpolate, restrict


class TestPygmg3d(unittest.TestCase):
    def test_interpolation_indices(self):
        print("1's are copied points from coarse grid")
        print("2's are 1st interpolation points")
        coarse_mesh = Mesh([3, 3, 3])
        for index in coarse_mesh.indices():
            coarse_mesh[index] = sum(index)
        coarse_mesh.print("Coarse mesh")
        fine_mesh = interpolate(coarse_mesh)

        fine_mesh.print("Fine mesh")

        print("Restricted Fine Mesh")
        restricted = restrict(fine_mesh)
        restricted.print("Restricted Mesh")

