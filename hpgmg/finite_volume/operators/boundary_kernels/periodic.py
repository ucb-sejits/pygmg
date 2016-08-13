from hpgmg.finite_volume.operators.boundary_kernels.kernel_generator import KernelGenerator

__author__ = 'nzhang-dev'


class PeriodicBoundary(KernelGenerator):
    def make_kernel(self, boundary):

        def kernel(level, mesh):
            for index in level.boundary_iterator(boundary):
                mesh[index] = mesh[index - level.interior_space * boundary]
        return kernel
