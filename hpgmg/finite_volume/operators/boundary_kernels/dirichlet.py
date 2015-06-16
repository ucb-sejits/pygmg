from hpgmg.finite_volume.operators.boundary_kernels.kernel_generator import KernelGenerator
import numpy as np
import functools
import operator

__author__ = 'nzhang-dev'

class DirichletBoundary(KernelGenerator):
    def make_kernel(self, boundary):
        multiplier = functools.reduce(operator.mul, (i for i in boundary if i != 0))

        def kernel(level, mesh):
            for index in level.boundary_iterator(boundary):
                #print(index, boundary)
                mesh[index] = multiplier * mesh[index - boundary]

        return kernel
