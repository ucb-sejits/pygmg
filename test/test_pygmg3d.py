from __future__ import print_function
from hpgmg.finite_volume.operators.stencil_operators import ConstantCoefficient7pt, add_constant_boundary, Sum6pt
from hpgmg.finite_volume.space import Space, Vector, Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.pymg3d import interpolate, restrict


class TestPygmg3d(unittest.TestCase):
    def test_interpolation_for_scaling(self):
        print("1's are copied points from coarse grid")
        print("2's are 1st interpolation points")
        coarse_mesh = Mesh([3, 3, 3])
        coarse_mesh.assign_to_all(1)
        coarse_mesh.print("Coarse mesh")
        fine_mesh = interpolate(coarse_mesh)

        self.assertTrue(
            all(map(lambda x: fine_mesh[x] == 1, fine_mesh.indices()))
        )
        # print("Restricted Fine Mesh")
        # restricted = restrict(fine_mesh)
        # restricted.print("Restricted Mesh")

    def test_interpolation_for_values(self):
        coarse_mesh = Mesh([3, 3, 3])

        for i in coarse_mesh.indices():
            if i[0] == 2:
                coarse_mesh[i] = 15
            else:
                coarse_mesh[i] = 1

        # running interpolation (the point of the test), and capturing output
        fine_mesh = interpolate(coarse_mesh)

        # constructing the expected fine mesh
        expected_fine_mesh = Mesh([5, 5, 5])
        for i in expected_fine_mesh.indices():
            if i[0] == 4:
                expected_fine_mesh[i] = 15
            elif i[0] == 3:
                expected_fine_mesh[i] = 8
            else:
                expected_fine_mesh[i] = 1

        # final test
        self.assertTrue(fine_mesh == expected_fine_mesh)

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
        pass

    def test_mul_and_div(self):
        s = Space(4, 4, 4)
        print(s)
        s /= 2
        print(s)
        self.assertEqual(s, Space(2, 2, 2))
        self.assertTrue(isinstance(s[0], int))

        v = Vector(4, 4, 4)
        v /= 2
        print(v)
        self.assertEqual(v, (2.0, 2.0, 2.0))
        self.assertTrue(isinstance(v[0], float))

        c = Coord(2, 2, 2)
        c /= 2
        self.assertTrue(isinstance(c[0], int))
        self.assertEqual(Coord(2, 2, 2)*2, Coord(4, 4, 4))
        print(c)

    def lin_fill(self, x):
        assert isinstance(x, Mesh)
        dim = x.shape[0]
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    x[i][j][k] = i*dim*dim + j*dim + k  # sam inverts ijk for some reason...this is the convention

    def test_apply_op(self):
        B = Sum6pt()
        s = Space(2, 2, 2)
        x = Mesh(s)
        self.lin_fill(x)  # value of each element is its ijk index
        x = add_constant_boundary(x, 1)  # adding a constant boundary of 1 to mesh
        print("Our mesh")
        print(x)
        result = B*x
        print("result of summing all 6 neighbors operator, boundaries ignored")
        print(result)
        result_val = 10 # 1+1+1+2+1+4, sum of neighbors of element (0,0,0) = 0 ie (1,1,1) in mesh w/ boundary
        for i in range(1, 3):
            for j in range(1, 3):
                for k in range(1, 3):
                    self.assertEqual(result[i][j][k], result_val)
                    result_val += 1


