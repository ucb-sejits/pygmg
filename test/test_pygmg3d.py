from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.pymg3d import interpolate, restrict


class TestPygmg3d(unittest.TestCase):
    def test_interpolation(self):
        print("1's are copied points from coarse grid")
        print("2's are 1st interpolation points")
        coarse_mesh = Mesh([3, 3, 3])
        for index in coarse_mesh.indices():
            coarse_mesh[index] = 1
        coarse_mesh.print("Coarse mesh")
        fine_mesh = interpolate(coarse_mesh)

        self.assertTrue(
            all(map(lambda x: fine_mesh[x] == 1, fine_mesh.indices()))
        )
        fine_mesh.print("Fine mesh")

        print("Restricted Fine Mesh")
        restricted = restrict(fine_mesh)
        restricted.print("Restricted Mesh")

    def test_restriction(self):
        fine_mesh = Mesh([5, 5, 5])
        for index in fine_mesh.indices():
            # fine_mesh[index] = sum(index)
            fine_mesh[index] = 1.0

        fine_mesh.print("FineMesh restriction test")

        coarse_mesh = restrict(fine_mesh)
        coarse_mesh.print("Coarse mesh restriction test")
        for index in coarse_mesh.indices():
            self.assertEqual(
                coarse_mesh[index],
                # sum(index*2),
                1,
                "index[{}] {} should be sum(index*2) {}".format(index, coarse_mesh[index], sum(index*2))
            )

    def test_interpolation_indices(self):
        print("1's are copied points from coarse grid")
        print("2's are 1st interpolation points")
        coarse_mesh = Mesh([3, 3, 3])
        for index in coarse_mesh.indices():
            coarse_mesh[index] = sum(index)
        coarse_mesh.print("Coarse mesh")
        fine_mesh = interpolate(coarse_mesh)

        fine_mesh.print("Fine mesh")


    def test_implementation(self):
