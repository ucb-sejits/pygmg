from __future__ import print_function
import unittest

import numpy as np
from math import sin, cos, pow

from hpgmg.finite_volume.problems.problem_fv import ProblemFV

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestProblemFV(unittest.TestCase):
    def setUp(self):
        self.a = 2.0 * np.pi
        self.power = 7.0


    def test_generators_for_other_dimensions(self):
        for d in range(1, 5):
            problem = ProblemFV(d)
            print("Dimensions {}".format(d))
            print("    {}".format(problem.expression))
            # following is crude test that we have the right number of terms
            self.assertEqual(d-1, problem.expression.__str__().count(')*('))


    @staticmethod
    def pow(value, exponent):
        return pow(value, exponent)

    def fv_evaluate_f(self, point):
        x, y, z = point
        return (
            pow(sin(self.a*x), self.power) *
            pow(sin(self.a*y), self.power) *
            pow(sin(self.a*z), self.power))

    def fv_correction(self, point, cell_size=1.0):
        x, y, z = point
        a = self.a
        p = self.power
        Fxx = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p-2)*pow(sin(a*y),p  )*pow(sin(a*z),p  )*pow(cos(a*x),2);
        Fyy = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p  )*pow(sin(a*y),p-2)*pow(sin(a*z),p  )*pow(cos(a*y),2);
        Fzz = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p-2)*pow(cos(a*z),2);

        return Fxx, Fyy, Fzz

    def compute_point_williams_method(self, point, add_4th_order_corrections=False, cell_size=1.0):
        a, p = 2.0 * np.pi, 7.0
        x, y, z = point
        from math import sin, cos

        F   =        pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )

        if add_4th_order_corrections:
            f_xx, f_yy, f_zz = self.fv_correction(point)
            F += (sum(f_xx, f_yy, f_zz) * (cell_size * cell_size) / 24.0)

        return F

    def test_basics(self):
        problem = ProblemFV(dimensions=3, cell_size=1/128.0, add_4th_order_correction=False)
        sample_point = (0.1, 0.1, 0.1)

        print("problem expression {}".format(problem.expression))
        print("symbols {}".format(problem.symbols))
        print("problem 2nd derivative {}".format(problem.get_derivative(dim=0, derivative=2)))

        func = problem.get_func(problem.expression, ("x0", "x1", "x2"))

        pygmg_value = func(*sample_point)
        print("pygmg_value {}".format(pygmg_value))

        williams_value = self.compute_point_williams_method(sample_point, add_4th_order_corrections=False)
        print("williams_value {}".format(williams_value))

        func2 = problem.get_func(problem.get_derivative(dim=0, derivative=2), ("x0", "x1", "x2"))
        print("sample point {}".format(func2(*sample_point)))

        a, p = 2.0 * np.pi, 7.0
        x, y, z = sample_point
        from math import sin, cos

        u_via_sams_method = -a*a*p*sin(a*x)**p * sin(a*y)**p * sin(a*z)**p + \
                    a*a*p*(p-1)*sin(a*x)**(p-2)*sin(a*y)**p * sin(a*z)**p * cos(a*x)**2



        print("sam's result {}".format(u_via_sams_method))