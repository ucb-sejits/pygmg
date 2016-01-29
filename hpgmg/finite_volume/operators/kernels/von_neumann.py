from __future__ import print_function
from snowflake.nodes import StencilComponent, SparseWeightArray
from snowflake.vector import Vector
from hpgmg.finite_volume.operators.kernels.base_kernel import BaseKernel

__author__ = 'nzhang-dev'


class VonNeumannStencil(BaseKernel):
    def __init__(self, dimensions, a, b, h2inv):
        self.dimensions = dimensions
        self.b = b
        self.a = a
        self.h2inv = h2inv


class VariableCoefficientVonNeumannStencil(VonNeumannStencil):
    def get_stencil(self):
        alpha_component = StencilComponent('alpha', SparseWeightArray({(0,)*self.dimensions: 1}))
        mesh_primary = StencilComponent('mesh', SparseWeightArray({(0,)*self.dimensions: 1}))
        scale_factor = self.a * alpha_component * mesh_primary - self.b * self.h2inv
        forward_facing_components = []
        backward_facing_components = []
        zero_vec = Vector.zero_vector(self.dimensions)
        for dim in range(self.dimensions):
            beta_back = StencilComponent('beta_{}'.format(dim), SparseWeightArray({zero_vec: 1}))
            mesh_back = StencilComponent('mesh', SparseWeightArray({
                -Vector.unit_vector(dim, self.dimensions): 1,
                zero_vec: -1
            }))
            beta_forwards = StencilComponent('beta_{}'.format(dim), SparseWeightArray({
                Vector.unit_vector(dim, self.dimensions): 1
            }))
            mesh_forwards = StencilComponent('mesh',SparseWeightArray({
                Vector.unit_vector(dim, self.dimensions): 1,
                zero_vec: -1
            }))
            forward_facing_components.append(beta_back * mesh_back)
            backward_facing_components.append(beta_forwards * mesh_forwards)
        total = scale_factor * sum(forward_facing_components + backward_facing_components)
        return total

class ConstantCoefficientVonNeumannStencil(VonNeumannStencil):
    def get_stencil(self):
        #print(self.a, self.b, self.h2inv)
        a_component = self.a * StencilComponent('mesh',
                                                SparseWeightArray({Vector.zero_vector(self.dimensions): 1}))
        von_neumann_points = list(Vector.von_neumann_vectors(self.dimensions, radius=1, closed=False))
        weights = {point: 1 for point in von_neumann_points}
        weights[Vector.zero_vector(self.dimensions)] = -len(von_neumann_points)
        b_component = self.b * self.h2inv * StencilComponent('mesh', SparseWeightArray(weights))
        return a_component - b_component