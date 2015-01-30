from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.pymg3d import interpolate
from hpgmg.finite_volume.space import Space, Coord


class TestMesh(unittest.TestCase):
    def test_coordinate_access(self):
        mesh = Mesh([3, 3, 3])
        mesh.fill(1)

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord], mesh[coord])

    def test_print(self):
        mesh = Mesh(Space(3,3,3))
        mesh.fill(2)
        mesh.print()

