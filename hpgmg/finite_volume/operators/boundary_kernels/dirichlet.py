from ctree.frontend import dump
from snowflake.nodes import StencilComponent, SparseWeightArray, Stencil
from snowflake.vector import Vector
from hpgmg.finite_volume.operators.boundary_kernels.kernel_generator import KernelGenerator
import numpy as np
import functools
import operator

__author__ = 'nzhang-dev'


class DirichletBoundary(KernelGenerator):
    """
    returns a pointer to a function that computes the value on
    the boundary.  each higher order boundary
    where O(corner) > O(edge) > O(face) ...
    is the opposite sign of its immediately lower boundary
    kernel.multiplier in make_kernel does this
    """
    def make_kernel(self, boundary):

        def kernel(level, mesh):
            for index in level.boundary_iterator(boundary):
                mesh[index] = kernel.multiplier * mesh[index - boundary]
        kernel.multiplier = (-1.0)**sum(abs(i) for i in boundary)
        return kernel

    def get_stencil(self, boundary):
        #print(boundary)
        vec = -1*Vector(boundary)
        weight = (-1.0)**sum(abs(i) for i in boundary)
        #print(vec, weight)
        component = StencilComponent(
            'mesh',
            SparseWeightArray(
                {
                    vec: weight
                }
            )
        )
        #print component.weights.vectors
        return Stencil(component, 'mesh', [
            (-1, 0, 1) if bound == 1 else (0, 1, 1) if bound == -1 else (1, -1, 1)
            for bound in boundary
        ])

