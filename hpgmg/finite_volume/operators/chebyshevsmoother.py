from __future__ import print_function

__author__ = 'Shiv Sundram shivsundram@berkeley.edu U.C. Berkeley, shivsundram@lbl.gov LBNL'

# Based on Yousef Saad's Iterative Methods for Sparse Linear Algebra, Algorithm 12.1, page 399
# with algorithmic corrections provided by Sam Williams

class ChebyshevSmoother(object):
    def __init__(self, op, degree=4, iterations=10):
        """

        :param op:
        :param use_l1_jacobi:
        :param iterations:
        :return:
        """
        self.operator = op
        self.iterations = 7
        self.degree = 4

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
        chebyshev_c2 = [float] * self.degree  # + c2*(b-Ax_n)
        chebyshev_c1[0] = 0.0
        chebyshev_c2[0] = 1/theta
        for s in range (1, self.degree):   # generate chebyshev polynomial coefficients
            rho_nm1 = rho_n
            rho_n = 1.0/(2.0*sigma - rho_nm1)
            chebyshev_c1[s] = rho_n*rho_nm1
            chebyshev_c2[s] = rho_n*2.0/delta

        for s in range(self.degree*self.iterations):  # need to store 2 prev src meshes
            if (s & 1) == 0:
                working_source = mesh_to_smooth
                working_source_prev = level.temp
                working_target = level.temp
            else:
                working_source = level.temp
                working_source_prev = mesh_to_smooth
                working_target = mesh_to_smooth
            c1 = chebyshev_c1[s % self.degree]
            c2 = chebyshev_c2[s % self.degree]

            lambda_mesh = level.d_inverse

            need_copy = False
            for index in level.interior_points():
                Ax = self.operator.apply_op(rhs_mesh, index, level)
                b = rhs_mesh[index]
                working_target[index] = mesh_to_smooth[index] + (
                    c1 * (working_source[index] - working_source_prev[index]) +
                    c2 * lambda_mesh[index] * (b - Ax)
                )
                print(lambda_mesh[index], b, Ax, )
            need_copy = not need_copy

        if need_copy:
            level.shift_mesh(mesh_to_smooth, 1.0, level.temp)


if __name__ == '__main__':
    import hpgmg.finite_volume.simple_hpgmg as simple_hpgmg

    solver = simple_hpgmg.SimpleMultigridSolver.get_solver("0 -sm c ".split())

    #assert isinstance(solver.smoother, ChebyshevSmoother), "solver.smoother {} is not a ChebyshevSmoother".format(
    #    solver.smoother)


    base_level = solver.fine_level
    mesh = base_level.cell_values
    for point in mesh.indices():
        mesh[point] = sum(list(point))
    mesh.print("mesh")

    solver.smoother.smooth(base_level, base_level.cell_values, base_level.exact_solution)
    base_level.cell_values.print("smoothed mesh")
