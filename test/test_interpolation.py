from __future__ import print_function
import numpy
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest
from hpgmg.finite_volume.operators.interpolation import InterpolatorPQ, InterpolatorPC
from hpgmg.finite_volume.space import Space, Coord, Vector


class TestInterpolationPQ(unittest.TestCase):
    def test_interpolation_pq_neighbor_stuff(self):
        solver = SimpleMultigridSolver.get_solver("3")
        interpolator = InterpolatorPQ(solver, 1.0)

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
        solver = SimpleMultigridSolver.get_solver("3")
        interpolator = InterpolatorPQ(solver, 1.0)

        all_even_index = Coord(0, 0, 0)
        self.assertEqual(InterpolatorPQ.compute_neighbor_index(all_even_index), 0, "all even index is category 0")
        k_odd = Coord(10, 1022, 7)
        self.assertEqual(InterpolatorPQ.compute_neighbor_index(k_odd), 1, "if just k is odd then category 1")
        all_odd = Coord(3, 5, 7)
        self.assertEqual(InterpolatorPQ.compute_neighbor_index(all_odd), 7, "if all are odd then category 7")

        coefficient_0, offsets_0 = interpolator.convolution[0]

        all_even_offset = offsets_0[InterpolatorPQ.compute_neighbor_index(all_even_index)]
        k_odd_offset = offsets_0[InterpolatorPQ.compute_neighbor_index(k_odd)]

        self.assertEqual(all_even_offset.i, k_odd_offset.i)
        self.assertEqual(all_even_offset.j, k_odd_offset.j)
        self.assertNotEqual(all_even_offset.k, k_odd_offset.k)

    def test_interpolator_convolution_values(self):

        def compute_neighbor_index(vector):
            return (vector.i % 2) * 4 + (vector.j % 2) * 2 + (vector.k % 2)

        expected_index = 0
        for index in Space(2, 2, 2).points:
            self.assertEqual(expected_index, compute_neighbor_index(index))
            print("{:3d}{:3d}{:3d}=>{:4d}  ".format(index.i, index.j, index.k, compute_neighbor_index(index)))
            expected_index += 1

        solver = SimpleMultigridSolver.get_solver("3")
        interpolator = InterpolatorPQ(solver, 1.0)
        print("nd {}".format([(x, i) for x, i in enumerate(interpolator.neighbor_directions)]))
        print(interpolator.convolution[0])
        print(interpolator.convolution[1])


class TestInterpolationPC(unittest.TestCase):
    def test_interpolation_on_uniform_space(self):
        solver = SimpleMultigridSolver.get_solver(["3", "--dimensions", "2"])
        level = solver.fine_level
        coarser_level = level.make_coarser_level()

        for index in coarser_level.indices():
            coarser_level.cell_values[index] = sum(index-coarser_level.ghost_zone)
        # coarser_level.fill_mesh(coarser_level.cell_values, 1.0)
        level.fill_mesh(level.cell_values, 0.0)

        self.assertIsInstance(solver.interpolator, InterpolatorPC)

        solver.interpolator.interpolate(level, level.cell_values, coarser_level.cell_values)

        coarser_level.cell_values.print("coarse cell values")
        level.cell_values.print("finer cell values")

        for index in level.interior_points():
            expected = sum((index - level.ghost_zone) // 2)
            self.assertEqual(
                level.cell_values[index], expected,
                "finer_mesh[{}] is {} should be {}".format(index, level.cell_values[index], expected)
            )

    def test_interpolation_on_singularity(self):
        solver = SimpleMultigridSolver.get_solver(["2", "--dimensions", "3"])
        level = solver.fine_level
        coarser_level = level.make_coarser_level()

        mesh = coarser_level.cell_values
        coarser_level.fill_mesh(mesh, 0.0)
        mesh[Coord(1, 1, 1)] = 1.0

        finer_mesh = level.cell_values

        interpolator = InterpolatorPC(solver, 0.0)

        interpolator.interpolate(level, finer_mesh, mesh)

        finer_mesh.print("Finer mesh")

        for point in finer_mesh.indices():
            expected = 1.0 if 0 < point.i < 3 and 0 < point.j < 3 and 0 < point.k < 3 else 0.0
            self.assertEqual(
                finer_mesh[point], expected,
                "finer_mesh[{}] is {} should be {}".format(point, finer_mesh[point], expected)
            )


class TestInterpolationPQCoefficients(unittest.TestCase):
    """
    attempt to divine the pq coefficients
    """
    def test_pq(self):
        print(len([
            0.421875,
            0.140625,
            0.140625,
            0.046875,
            0.140625,
            0.046875,
            0.046875,
            0.015625,
        ]))

        mesh = Mesh((3, 3, 3))
        mesh[Coord(-1, -1, -1) + Coord(1, 1, 1)] = -27.0
        mesh[Coord(0, -1, -1) + Coord(1, 1, 1)] = 270.0
        mesh[Coord(+1, -1, -1) + Coord(1, 1, 1)] = 45.0
        mesh[Coord(-1, 0, -1) + Coord(1, 1, 1)] = 270.0
        mesh[Coord(0, 0, -1) + Coord(1, 1, 1)] = -2700.0
        mesh[Coord(+1, 0, -1) + Coord(1, 1, 1)] = -450.0
        mesh[Coord(-1, +1, -1) + Coord(1, 1, 1)] = 45.0
        mesh[Coord(0, +1, -1) + Coord(1, 1, 1)] = -450.0
        mesh[Coord(+1, +1, -1) + Coord(1, 1, 1)] = -75.0
        mesh[Coord(-1, -1, 0) + Coord(1, 1, 1)] = 270.0
        mesh[Coord(0, -1, 0) + Coord(1, 1, 1)] = -2700.0
        mesh[Coord(+1, -1, 0) + Coord(1, 1, 1)] = -450.0
        mesh[Coord(-1, 0, 0) + Coord(1, 1, 1)] = -2700.0
        mesh[Coord(0, 0, 0) + Coord(1, 1, 1)] = 27000.0
        mesh[Coord(+1, 0, 0) + Coord(1, 1, 1)] = 4500.0
        mesh[Coord(-1, +1, 0) + Coord(1, 1, 1)] = -450.0
        mesh[Coord(0, +1, 0) + Coord(1, 1, 1)] = 4500.0
        mesh[Coord(+1, +1, 0) + Coord(1, 1, 1)] = 750.0
        mesh[Coord(-1, -1, +1) + Coord(1, 1, 1)] = 45.0
        mesh[Coord(0, -1, +1) + Coord(1, 1, 1)] = -450.0
        mesh[Coord(+1, -1, +1) + Coord(1, 1, 1)] = -75.0
        mesh[Coord(-1, 0, +1) + Coord(1, 1, 1)] = -450.0
        mesh[Coord(0, 0, +1) + Coord(1, 1, 1)] = 4500.0
        mesh[Coord(+1, 0, +1) + Coord(1, 1, 1)] = 750.0
        mesh[Coord(-1, +1, +1) + Coord(1, 1, 1)] = -75.0
        mesh[Coord(0, +1, +1) + Coord(1, 1, 1)] = 750.0
        mesh[Coord(+1, +1, +1) + Coord(1, 1, 1)] = 125.0

        mesh.print("convolution")
        print(numpy.sum(mesh))

