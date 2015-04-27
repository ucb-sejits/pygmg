from __future__ import print_function
from hpgmg.finite_volume.operators.jacobi_smoother import JacobiSmoother
from hpgmg.finite_volume.operators.stencil_von_neumann_r1 import StencilVonNeumannR1
import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest

from hpgmg.finite_volume.space import Space
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.pymg3d import interpolate



class TestApplyOp(unittest.TestCase):


    mesh = Mesh(2,2)
    def constructMatrix(self, x):
        length = x.space[0]
        matrix = Mesh((length**2, length**2))
        matrix.fill(0)
        for i in range(0, length):
            for j in range(0, length):
                ij = i*length +j
                if not  ((0<i and i<length-1) and (0<j and j<length-1) ):
                    matrix[ij][ij] = 1
                else:
                    neighbors = neighbors2d(i, j)
                    #print(neighbors)
                    for neighbor in neighbors:
                        ij1 = length*neighbor.i + neighbor.j
                        matrix[ij][ij1] = 1
                    matrix[ij][ij] = -4
        return matrix



    #construct test level

    solver = simple_hpgmg.SimpleMultigridSolver.get_solver("1".split())
    base_level = solver.fine_level
    mesh = base_level.cell_values
    for point in mesh.indices():
        mesh[point] = sum(list(point))
    mesh.print("mesh")

    solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
    base_level.cell_values.print("smoothed mesh")
"""
    def test2dApplyOp(self):
        mesh = Mesh([3, 3])
        mesh.fill(1)
        print(mesh)
        op = StencilVonNeumannR1(solver=self)
        J = JacobiSmoother(op, True, 1)
"""


