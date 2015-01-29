from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.space import Coord


class TestMesh(unittest.TestCase):
    def test_coordinate_access(self):
        mesh = Mesh([3, 3, 3])
        mesh.fill(1)

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], 1)
            self.assertEqual(mesh[coord.to_tuple()], 1)
            self.assertEqual(mesh[coord], mesh[coord.to_tuple()])

        for coord in mesh.indices():
            mesh[coord] = coord.volume()

        for coord in mesh.indices():
            self.assertEqual(mesh[coord], coord.volume())
            self.assertEqual(mesh[coord.to_tuple()], coord.volume())

    def test_print(self):
        mesh = Mesh([3, 3, 3])
        for index in mesh.indices():
            mesh[index] = index.volume()

        mesh.print()

    def test_red(self):
        print("red 1 black 2")
        mesh = Mesh([3, 3, 3])
        for index in mesh.indices():
            mesh[index] = 1 if index.is_red() else 2

        mesh.print()

    def test_interpolation_indices(self):
        print("1's are copied points from coarse grid")
        print("2's are 1st interpolation points")
        coarse_space = Coord(3, 3, 3)
        mesh = Mesh(coarse_space.double().to_tuple())
        mesh.fill(0)

        for index in (coarse_space+1).foreach():
            target = index * 2
            mesh[target] = 1
        for index in coarse_space.foreach():
            interpolation_point = index * 2 + 1
            mesh[interpolation_point] = 2

        mesh.print()

