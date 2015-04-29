from __future__ import print_function
from hpgmg.finite_volume.operators.base_operator import BaseOperator
from hpgmg.finite_volume.operators.smoother import Smoother

__author__ = 'Shiv Sundram shivsundram@berkeley.edu U.C. Berkeley, shivsundram@lbl.gov LBNL'

# Based on Yousef Saad's Iterative Methods for Sparse Linear Algebra, Algorithm 12.1, page 399
# with algorithmic corrections provided by Sam Williams


class ChebyshevSmoother(Smoother):
    def __init__(self, op, degree=4, iterations=10):
        """
        :param op:
        :param degree:
        :param iterations:
        :return:
        """
        assert isinstance(op, BaseOperator)
        assert isinstance(degree, int)
        assert isinstance(iterations, int)

        self.operator = op
        self.iterations = iterations
        self.degree = degree

    def smooth(self, level, mesh_to_smooth, rhs_mesh):
        """

        :param level: the level being smoothed
        :param mesh_to_smooth:
        :param rhs_mesh:
        :return:
        """
        beta = 1.000*level.dominant_eigen_value_of_d_inv_a
        alpha = 0.125000*beta
        theta = 0.5*(beta+alpha)		# center of the spectral ellipse
        delta = 0.5*(beta-alpha)		# major axis?
        sigma = theta/delta
        rho_n = 1/sigma			# rho_0
        chebyshev_c1 = [float] * self.degree  # + c1*(x_n-x_nm1) == rho_n*rho_nm1
        chebyshev_c2 = [float] * self.degree  # + c2*(b-a_x_n)
        chebyshev_c1[0] = 0.0
        chebyshev_c2[0] = 1/theta
        for s in range(1, self.degree):   # generate chebyshev polynomial coefficients
            rho_nm1 = rho_n
            rho_n = 1.0/(2.0*sigma - rho_nm1)
            chebyshev_c1[s] = rho_n*rho_nm1
            chebyshev_c2[s] = rho_n*2.0/delta

        self.operator.set_scale(level.h)

        need_copy = False
        for s in range(self.degree*self.iterations):  # need to store 2 prev src meshes
            if (s & 1) == 0:
                working_source = mesh_to_smooth
                working_source_prev = level.temp
                working_target = level.temp
            else:
                working_source = level.temp
                working_source_prev = mesh_to_smooth
                working_target = mesh_to_smooth

            level.solver.boundary_updater.apply(level, working_source)
            c1 = chebyshev_c1[s % self.degree]
            c2 = chebyshev_c2[s % self.degree]

            lambda_mesh = level.d_inverse

            for index in level.interior_points():
                a_x = self.operator.apply_op(working_source, index, level)
                b = rhs_mesh[index]
                working_target[index] = working_source[index] + (
                    c1 * (working_source[index] - working_source_prev[index]) +
                    c2 * lambda_mesh[index] * (b - a_x)
                )
            need_copy = not need_copy

        if need_copy:
            level.copy_mesh(mesh_to_smooth, level.temp)
