from __future__ import print_function
import copy
import inspect
from ctree.frontend import dump
from snowflake.nodes import StencilComponent, SparseWeightArray, Stencil, StencilGroup
from snowflake.utils import swap_variables
from snowflake.vector import Vector
from hpgmg.finite_volume import compiler

from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.smoother import Smoother
from hpgmg.finite_volume.operators.specializers.smooth_specializer import CSmoothSpecializer, OmpSmoothSpecializer
from hpgmg.finite_volume.operators.specializers.util import specialized_func_dispatcher, profile, time_this

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class JacobiSmoother(Smoother):
    def __init__(self, op, use_l1_jacobi=True, iterations=6):
        """

        :param op:
        :param use_l1_jacobi:
        :param iterations:
        :return:
        """
        assert(isinstance(op, BaseOperator))
        assert(isinstance(use_l1_jacobi, bool))
        assert(isinstance(iterations, int))
        assert iterations % 2 == 0

        self.operator = op
        self.use_l1_jacobi = use_l1_jacobi
        self.weight = 1.0 if use_l1_jacobi else 2.0/3.0
        self.iterations = iterations
        self.__kernels = {}
        self.__smooth_kernels = {}

    def get_stencil(self, level):
        #print("Rebuilding stencil")
        a_x = level.kernel.get_stencil()
        b = StencilComponent('rhs_mesh', SparseWeightArray({Vector.zero_vector(self.operator.dimensions): 1}))
        lambda_ref = StencilComponent('lambda_mesh', SparseWeightArray({Vector.zero_vector(self.operator.dimensions): 1}))
        working_ref = StencilComponent('mesh', SparseWeightArray({Vector.zero_vector(self.operator.dimensions): 1}))
        rhs = working_ref + (self.weight * lambda_ref * (b - a_x))
        return Stencil(rhs, 'out', ((1, -1, 1),) * self.operator.dimensions, primary_mesh='out')

    def get_kernel(self, level):
        if level in self.__kernels:
            return self.__kernels[level]
        stencil = self.get_stencil(level)
        kernel = self.__kernels[level] = compiler.compile(stencil)
        return kernel

    def get_smooth_stencil(self, level, iterations):
        stencil = self.get_stencil(level)
        boundary_kernels = copy.deepcopy(level.solver.boundary_updater.stencil_kernels)
        group = boundary_kernels + [stencil]
        copied = [swap_variables(s, {'out': 'mesh', 'mesh': 'out'}) for s in group]
        composed = []
        for i in range(iterations):
            if i % 2 == 0:
                composed.extend(copied)
            else:
                composed.extend(group)
        stencil_group = StencilGroup(composed)
        return stencil_group
        # return StencilGroup(boundary_kernels + [stencil])

    def get_smooth_kernel(self, level, iterations):
        if level in self.__smooth_kernels:
            return self.__smooth_kernels[level]
        stencil = self.get_smooth_stencil(level, iterations)
        kernel = self.__smooth_kernels[level] = compiler.compile(stencil)
        return kernel

    @time_this
    def std_smooth(self, level, mesh_to_smooth, rhs_mesh):
        """

        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        lambda_mesh = level.l1_inverse if self.use_l1_jacobi else level.d_inverse

        working_target, working_source = mesh_to_smooth, level.temp

        self.operator.set_scale(level.h)
        # print("\n\nSMOOTH PASS")
        for i in range(self.iterations):
            working_target, working_source = working_source, working_target
            level.solver.boundary_updater.apply(level, working_source)
            self.smooth_points(level, working_source, working_target, rhs_mesh, lambda_mesh)
        # print(mesh_to_smooth[:4, :4, :4])
        # raise Exception()

    # @time_this
    def kernel_smooth(self, level, mesh_to_smooth, rhs_mesh):
        """
        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        lambda_mesh = level.l1_inverse if self.use_l1_jacobi else level.d_inverse

        working_target, working_source = mesh_to_smooth, level.temp

        self.operator.set_scale(level.h)

        kernel = self.get_smooth_kernel(level, self.iterations)
        # for i in range(self.iterations):
        #     working_target, working_source = working_source, working_target
        #
        #     kernel(working_source, working_target, lambda_mesh, rhs_mesh)
        kernel(working_source, working_target, lambda_mesh, rhs_mesh)

    smooth = kernel_smooth

    @time_this
    @specialized_func_dispatcher({
        'c': CSmoothSpecializer,
        'omp': OmpSmoothSpecializer
    })
    def smooth_points(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        for index in level.interior_points():
            a_x = self.operator.apply_op(working_source, index, level)
            b = rhs_mesh[index]
            working_target[index] = working_source[index] + (self.weight * lambda_mesh[index] * (b - a_x))

    def kernel_smooth_points(self, level, working_source, working_target, rhs_mesh, lambda_mesh):
        if not self.operator.is_variable_coefficient:
            self.get_kernel(level)(working_target, lambda_mesh, working_source, rhs_mesh)
            #print(self.get_kernel(level).arg_spec)