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

    # noinspection PyPep8,PyPep8Naming
    def fv_correction(self, point):
        x, y, z = point
        a = self.a
        p = self.power
        Fxx = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p-2)*pow(sin(a*y),p  )*pow(sin(a*z),p  )*pow(cos(a*x),2)
        Fyy = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p  )*pow(sin(a*y),p-2)*pow(sin(a*z),p  )*pow(cos(a*y),2)
        Fzz = -a*a*p*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )  +  a*a*p*(p-1)*pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p-2)*pow(cos(a*z),2)

        return Fxx, Fyy, Fzz

    # noinspection PyPep8,PyPep8Naming
    def compute_point_williams_method(self, point, add_4th_order_corrections=False, cell_size=1.0):
        a, p = 2.0 * np.pi, 7.0
        x, y, z = point

        F = pow(sin(a*x),p  )*pow(sin(a*y),p  )*pow(sin(a*z),p  )

        if add_4th_order_corrections:
            f_xx, f_yy, f_zz = self.fv_correction(point)
            F += (sum([f_xx, f_yy, f_zz]) * (cell_size * cell_size) / 24.0)

        return F

    def test_basics_no_correction(self):
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

        self.assertAlmostEqual(pygmg_value, williams_value)

    def test_fv_correction_values(self):
        problem = ProblemFV(dimensions=3, cell_size=1/128.0, add_4th_order_correction=True)
        sample_point = (0.1, 0.1, 0.1)

        print("problem expression {}".format(problem.expression))
        print("symbols {}".format(problem.symbols))
        print("problem 2nd derivative {}".format(problem.get_derivative(dim=0, derivative=2)))
        print()
        print()
        williams_corrections = list(self.fv_correction(sample_point))

        for dim in range(3):
            print("correction for {} {}".format(dim, problem.second_derivatives[dim]))
        pygmg_corrections = [
            problem.get_func(problem.second_derivatives[dim], problem.symbols)(*sample_point)
            for dim in range(3)
        ]

        for dim in range(3):
            print("{:12.8f} -- {:12.8f}".format(williams_corrections[dim], pygmg_corrections[dim]))
            self.assertAlmostEqual(williams_corrections[dim], pygmg_corrections[dim])

    def test_basics_with_correction(self):
        problem = ProblemFV(dimensions=3, cell_size=1/128.0, add_4th_order_correction=True)
        sample_point = (0.1, 0.1, 0.1)

        print("problem expression {}".format(problem.expression))
        print("symbols {}".format(problem.symbols))
        print("problem 2nd derivative {}".format(problem.get_derivative(dim=0, derivative=2)))

        func = problem.get_func(problem.expression, ("x0", "x1", "x2"))

        pygmg_value = func(*sample_point)
        print("pygmg_value {}".format(pygmg_value))

        williams_value = self.compute_point_williams_method(sample_point,
                                                            add_4th_order_corrections=True, cell_size=1/128.0)
        print("williams_value {}".format(williams_value))

        self.assertAlmostEqual(pygmg_value, williams_value)
