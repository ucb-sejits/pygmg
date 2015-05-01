from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.space import Space
from hpgmg.finite_volume.mesh import Mesh


class TestMesh(unittest.TestCase):
    def test_coordinate_access(self):
        mesh = Mesh([3, 3, 3])
        mesh.fill(1)

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord], mesh[coord])

        for coord in mesh.indices():
            mesh[coord] = Space(coord).volume

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], Space(coord).volume)
            self.assertEqual(mesh[coord], Space(coord).volume)

    def test_print(self):
        mesh = Mesh(Space(3,3,3))
        mesh.fill(2)
        mesh.print()

    def test_eq_true(self):
        mesh_1 = Mesh([5, 5, 5])
        for i in mesh_1.indices():
            if i[0] == 4:
                mesh_1[i] = 15
            elif i[0] == 3:
                mesh_1[i] = 8
            else:
                mesh_1[i] = 1

        mesh_2 = Mesh([5, 5, 5])
        for i in mesh_2.indices():
            if i[0] == 4:
                mesh_2[i] = 15
            elif i[0] == 3:
                mesh_2[i] = 8
            else:
                mesh_2[i] = 1

        self.assertTrue(mesh_1 == mesh_2)

    def test_eq_false_value(self):
        mesh_1 = Mesh([5, 5, 5])
        for i in mesh_1.indices():
            if i[0] == 4:
                mesh_1[i] = 15
            elif i[0] == 3:
                mesh_1[i] = 8
            else:
                mesh_1[i] = 1

        mesh_2 = Mesh([5, 5, 5])
        for i in mesh_2.indices():
            if i[0] == 4:
                mesh_2[i] = 15
            elif i[0] == 3:
                mesh_2[i] = 8
            else:
                mesh_2[i] = 1

        mesh_2[(0, 0, 0)] = 0

        self.assertFalse(mesh_1 == mesh_2)

