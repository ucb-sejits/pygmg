from __future__ import print_function
import copy
import inspect
from ctree.frontend import dump
from snowflake.nodes import StencilComponent, SparseWeightArray, Stencil, StencilGroup, RectangularDomain, DomainUnion
from snowflake.utils import swap_variables
from snowflake.vector import Vector
import time
from hpgmg import finite_volume

from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.smoother import Smoother
from hpgmg.finite_volume.operators.specializers.smooth_specializer import CSmoothSpecializer, OmpSmoothSpecializer
from hpgmg.finite_volume.operators.specializers.inline_jit import partial_jit
from hpgmg.finite_volume.operators.specializers.smooth_specializer import CSmoothSpecializer, OmpSmoothSpecializer, \
    OclSmoothSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher, profile, time_this


class GSRBSmoother(Smoother):
    def __init__(self, op, iterations=6):
        """
        :param op:
        :param iterations:
        :return:
        """
        assert(isinstance(op, BaseOperator))
        assert(isinstance(iterations, int))
        assert iterations % 2 == 0

        self.operator = op
        self.iterations = iterations
        self.__kernels = {}
        self.__smooth_kernels = {}

    @staticmethod
    def create_red(ndim):
        # upper faces of cube at 1, 1, 1 and 2, 2, 2...
        output = []
        for dim in range(ndim):
            output.append(RectangularDomain([(1, -1, 2) if d != dim else (2, -1, 2) for d in range(ndim)]))
        output.append(RectangularDomain([(2, -1, 2)]*ndim))
        return DomainUnion(output)

    @staticmethod
    def create_black(ndim):
        output = []
        start = Vector([1,]*ndim)
        primary = RectangularDomain([(1, -1, 2)]*ndim)
        # start from 1, 1, 1 ... and add 0, 0, ... 1, 0, .... 1, 0... 0, where 1-norm(delta) == 2
        for i in range(ndim):
            for j in range(i+1, ndim):
                delta = [0]*ndim
                delta[i] = delta[j] = 1
                low = start + delta
                output.append(RectangularDomain([(l, -1, 2) for l in low]))
        return DomainUnion(output + [primary])

    def get_stencil(self, level):
        #print("Rebuilding stencil")
        a_x = level.kernel.get_stencil()
        b = StencilComponent('rhs_mesh', SparseWeightArray({Vector.zero_vector(self.operator.dimensions): 1}))
        lambda_ref = StencilComponent('lambda_mesh', SparseWeightArray({Vector.zero_vector(self.operator.dimensions): 1}))
        working_ref = StencilComponent('mesh', SparseWeightArray({Vector.zero_vector(self.operator.dimensions): 1}))
        rhs = working_ref + lambda_ref * (b - a_x)
        stencils = StencilGroup([
            Stencil(rhs, 'mesh', self.create_red(self.operator.dimensions), primary_mesh='mesh'),
            Stencil(rhs, 'mesh', self.create_black(self.operator.dimensions), primary_mesh='mesh')
        ])
        return stencils

    def get_kernel(self, level):
        if level in self.__kernels:
            return self.__kernels[level]
        stencil = self.get_stencil(level)
        kernel = self.__kernels[level] = finite_volume.compiler.compile(stencil)
        return kernel

    def get_smooth_stencil(self, level, iterations):
        stencil = self.get_stencil(level).body
        boundary_kernels = copy.deepcopy(level.solver.boundary_updater.stencil_kernels)
        group = boundary_kernels + [stencil[0]] + boundary_kernels + [stencil[1]]
        stencil_group = StencilGroup(group*(iterations//2))
        return stencil_group

    def get_smooth_kernel(self, level, iterations):
        if level in self.__smooth_kernels:
            return self.__smooth_kernels[level]
        stencil = self.get_smooth_stencil(level, iterations)
        kernel = self.__smooth_kernels[level] = finite_volume.compiler.compile(stencil)
        return kernel

    @time_this
    def kernel_smooth(self, level, mesh_to_smooth, rhs_mesh):
        """
        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        lambda_mesh = level.d_inverse

        working_target, working_source = mesh_to_smooth, level.temp

        self.operator.set_scale(level.h)

        kernel = self.get_smooth_kernel(level, self.iterations)
        if finite_volume.CONFIG.variable_coefficient:
            #'mesh', 'out', 'alpha', 'beta_0', 'beta_1', 'beta_2', 'lambda_mesh', 'rhs_mesh'
            params = [working_target, level.alpha] + level.beta_face_values + [lambda_mesh, rhs_mesh]
            kernel(*params)
        else:
            kernel(working_target, lambda_mesh, rhs_mesh)

    smooth = kernel_smooth

    @time_this
    @specialized_func_dispatcher({
        'c': CSmoothSpecializer,
        'omp': OmpSmoothSpecializer,
        'ocl': OclSmoothSpecializer
    })
    def smooth_points(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        for index in level.interior_points():
            a_x = self.operator.apply_op(working_source, index, level)
            b = rhs_mesh[index]
            working_target[index] = working_source[index] + (self.weight * lambda_mesh[index] * (b - a_x))