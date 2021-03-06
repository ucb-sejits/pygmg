from __future__ import print_function
from abc import ABCMeta, abstractmethod
from hpgmg.finite_volume.operators.specializers.interpolate_specializer import CInterpolateSpecializer, \
    OclInterpolateSpecializer
from hpgmg.finite_volume.operators.specializers.util import time_this, specialized_func_dispatcher

from hpgmg.finite_volume.space import Space, Coord
import sympy
import numpy as np
from sympy import *
from sympy.abc import x
import itertools


__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Interpolator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def interpolate(self, target_level, target_mesh, source_mesh, ):
        pass


class InterpolatorPC(Interpolator):
    """
    interpolates by sampling
    """
    def __init__(self, solver, pre_scale):
        self.solver = solver
        self.dimensions = solver.dimensions
        self.pre_scale = pre_scale
        self.__kernels = {}

    @time_this
    @specialized_func_dispatcher({
        'c': CInterpolateSpecializer,
        'omp': CInterpolateSpecializer,
        'ocl': OclInterpolateSpecializer
    })
    def interpolate(self, target_level, target_mesh, source_mesh):
        for target_index in target_level.interior_points():
            source_index = ((target_index - target_level.ghost_zone) // 2) + target_level.ghost_zone
            target_mesh[target_index] *= self.pre_scale
            target_mesh[target_index] += source_mesh[source_index]

class InterpolatorNDPC(Interpolator):
    """
    interpolates in order n by sampling
    """
    def __init__(self, solver, pre_scale):
        self.solver = solver
        self.dimensions = solver.dimensions
        self.pre_scale = pre_scale

    @time_this
    @specialized_func_dispatcher({
        'c': CInterpolateSpecializer,
        'omp': CInterpolateSpecializer
    })
    def interpolate_std(self, target_level, target_mesh, source_mesh):
        for target_index in target_level.interior_points():
            source_index = ((target_index - target_level.ghost_zone) // 2) + target_level.ghost_zone
            target_mesh[target_index] *= self.pre_scale
            target_mesh[target_index] += source_mesh[source_index]

    # def get_stencil(self, level):
    #     stencils = []
    #     iter_range = level.interior_points().ranges
    #     for offset in itertools.product((0, 1), repeat=self.dimensions):
    #         irange = tuple(
    #             ((l - g) // 2 + g, (h - g) // 2 + g) for g, (l, h) in zip(level.ghost_zone, iter_range)
    #         )
    #         target_component = StencilComponent(
    #             'target_mesh',
    #             SparseWeightArray({Vector.zero_vector(self.dimensions): 1})
    #         ) * self.pre_scale
    #         source_component = StencilComponent(
    #             'source_mesh',
    #             SparseWeightArray({Vector.zero_vector(self.dimensions): 1})
    #         )
    #         stencil = ScalingStencil(
    #             target_component + source_component,
    #             'target_mesh',
    #             irange,
    #             level.ghost_zone,
    #             level.ghost_zone + offset,
    #             (2,) * self.dimensions
    #         )
    #         stencils.append(stencil)
    #     return StencilGroup(stencils)
    #
    # def get_kernel(self, level):
    #     if level in self.__kernels:
    #         return self.__kernels[level]
    #     kern = self.__kernels[level] = compiler.compile(self.get_stencil(level))
    #     return kern
    #
    #
    # def interpolate_kernel(self, target_level, target_mesh, source_mesh):
    #     kern = self.get_kernel(target_level)
    #     kern(target_mesh, source_mesh)
    #     raise Exception()

    interpolate = interpolate_std




def to_coeff_array(expressions):
    symbols = list(sorted(set(itertools.chain(*[i.free_symbols for i in expressions])), key=str))
    coeff_array = np.zeros((len(expressions), len(symbols)))
    for index, expression in enumerate(expressions):
        coeff_dict = expression.as_coefficients_dict()
        coeff_array[index] = [coeff_dict[sym] for sym in symbols]
    return coeff_array

class InterpolatorPQ(Interpolator):

    def __init__(self, solver, prescale=1.0):
        self.solver = solver
        self.pre_scale = prescale

        assert self.solver.dimensions == 3, "PQ interpolator currently only works in 3d"
        #
        # the interpolation is based on by associating a coefficient with a neighbor of
        # a position in the grid being interpolated

        #evaluate 2d polynomial u(x)=ax^2+bx+c algebraically at u(-1),u(0),u(1)
        #and store results in l
        # creating a system of equations where u(-1),u(0),u(1) are in terms of a,b, and c
        a, b, c= symbols("a, b, c")
        u = Poly(a*x**2 + b*x + c, x)
        l = [u(i) for i in range(-1,2)]

        #convert system of equations to matrix and "solve" by inverting matrix
        #to express a,b,c in terms of  u(-1), u(0) and u(1)
        B = np.linalg.inv(to_coeff_array(l))

        #express u(1/4) in terms of a, b and c
        uonefourth = to_coeff_array([u(.25)])[0]

        #express u(1/4) in terms of (-1), u(0) and u(1) via substitution
        #coeffs represent the coefficients in corresponding linear combination
        #and define the interpolating stencil
        coeffs=np.zeros(3)
        for i in range(3):
            coeffs += B[i]*uonefourth[i]
            #print(B[i], uonefourth[i])
        coeffs = [coeff*32 for coeff in coeffs]

        #compose coeffs to generate coefficients for higher order 3d interpolating stencil
        tup = [-1,0,1]
        self.convolution = []
        for i in itertools.product(tup, tup, tup):
            coord = Coord(i)
            shifted = [cc+1 for cc in coord] #convert from -1 indexed to zero indexed coordinate
            entry =  coeffs[shifted[0]]*coeffs[shifted[1]]*coeffs[shifted[2]], coord
            #print (entry)
            self.convolution.append(entry)

        # but we want the given neighbor coord to be either backward or forward looking depending on whether
        # the grid index for that dimension is odd or even respectively
        # to do this we we convert each coord above into an array of 8 coords where each index is flipped
        # depending on the evenness of the source interpolation point

        self.neighbor_directions = [InterpolatorPQ.compute_neighbor_direction(point) for point in Space(2, 2, 2).points]
        self.convolution = [
            (row[0], [row[1] * neighbor_direction for neighbor_direction in self.neighbor_directions])
            for row in self.convolution
        ]
        self.one_over_32_cubed = 1.0/(32**3)

    @staticmethod
    def compute_neighbor_direction(coord):
        return Coord([1 if c % 2 == 0 else -1 for c in coord])

    @staticmethod
    def compute_neighbor_index(vector):
            return (vector.i % 2) * 4 + (vector.j % 2) * 2 + (vector.k % 2)
    @time_this
    @specialized_func_dispatcher({
        'c': CInterpolateSpecializer,
    })
    def interpolate(self, target_level, target_mesh, source_level, source_mesh, ):
        for target_index in target_level.interior_points():
            source_index = target_index // 2
            target_mesh[target_index] *= self.pre_scale
            oddness_index = InterpolatorPQ.compute_neighbor_index(target_index)

            accumulator = 0
            for coefficient, neighbor_index_offsets in self.convolution:
                accumulator += coefficient * source_mesh[source_index + neighbor_index_offsets[oddness_index]]

            target_mesh[target_index] += self.one_over_32_cubed * accumulator
