from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.space import Space, Coord
from hpgmg.finite_volume.operators.nd_interpolate_generator import NDInterpolateGenerator
from hpgmg.finite_volume.operators.interpolation import InterpolatorPQ, InterpolatorPC, InterpolatorND
import unittest

__author__ = 'Shiv Sundram shivsundram@berkeley.edu U.C. Berkeley, shivsundram@lbl.gov, LBNL'

class TestInterpolationND(unittest.TestCase):
    def test_interpolation(self):
        solver = SimpleMultigridSolver.get_solver(["2", "--dimensions", "2"])
        level = solver.fine_level
        coarser_level = level.make_coarser_level()

        mesh = coarser_level.cell_values
        coarser_level.fill_mesh(mesh, 2.0)
        mesh.print("coarser")
        level.fill_mesh(level.cell_values, 0.0)
        finer_mesh = level.cell_values

        finer_mesh.print("Finer mesh")
        interpolator = InterpolatorND(solver, 0.0)

        interpolator.interpolate(level, finer_mesh, mesh)

        finer_mesh.print("Finer mesh")
