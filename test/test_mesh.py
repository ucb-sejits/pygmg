from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.pymg3d import interpolate


class TestMesh(unittest.TestCase):
    def test_coordinate_access(self):
        mesh = Mesh([3, 3, 3])
        mesh.fill(1)

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord], mesh[coord])

        for coord in mesh.indices():
            mesh[coord] = coord.volume()

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], coord.volume())
            self.assertEqual(mesh[coord.to_tuple()], coord.volume())

    def test_print(self):
        mesh = Mesh(Space(3,3,3))
        mesh.fill(2)
        mesh.print()

