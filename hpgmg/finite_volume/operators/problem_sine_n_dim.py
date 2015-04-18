from __future__ import print_function
from hpgmg.finite_volume.mesh import Mesh

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import numpy as np
from hpgmg.finite_volume.operators.problem import Problem
from hpgmg.finite_volume.operators.problem_sine import SineProblem
from hpgmg.finite_volume.space import Vector, Space

import sympy


class SineProblemND(Problem):
    """
    computes the value of the function u(...) in an arbitrary number of dimensions
    e.g.
    u(x) = sin(c1*x)^power + sin(c2*x)^power  # for the 1d
    u(x,y) = sin(c1*x)^power*sin(c1*y)^power + sin(c2*x)^power*sin(c2*y)^power  # for the 2d
    and so forth
    Also computes the first and second derivative of u with respect to each dimension
    Computation is done symbolically then coerced into a faster executing lambda instruction
    """

    def __init__(self, dimensions=3):
        expr = None
        dimension_names = ['x', 'y', 'z'][:dimensions]
        if dimensions > len(dimension_names):
            dimension_names += [chr(x+ord('a')) for x in range(dimensions-len(dimension_names))]

        dimension_symbols = []
        for dimension_name in dimension_names:
            declaration = "{0} = sympy.Symbol('{0}')".format(dimension_name)
            print("declaration {}".format(declaration))
            exec declaration
            dimension_symbols.append(sympy.Symbol(dimension_name))

        first_terms = ["(sympy.sin(c1*{})**power)".format(sym) for sym in dimension_symbols]
        first_product = " * ".join(first_terms)

        second_terms = ["(sympy.sin(c2*{})**power)".format(sym) for sym in dimension_symbols]
        second_product = " * ".join(second_terms)
        text_expression = "expr = " + first_product + " + " + second_product

        print("first terms {}".format(first_terms))
        print("second terms {}".format(second_terms))
        print("Expression {}".format(text_expression))

        c1, c2 = 2.0 * np.pi, 6.0 * np.pi
        power = 13.0
        exec text_expression
        print("expression {}".format(expr))

        first_derivatives = []
        second_derivatives = []
        for dimension_symbol in dimension_symbols:
            first_derivatives.append(
                sympy.diff(expr, dimension_symbol)
            )
            second_derivatives.append(
                sympy.diff(expr, dimension_symbol, 2)
            )

        print("first derivatives")
        for first_derivative in first_derivatives:
            print(first_derivatives)

        print("second derivatives")
        for second_derivative in second_derivatives:
            print(second_derivatives)

        self.u_function = sympy.lambdify(dimension_symbols, expr)
        self.u_first_derivatives = [
            sympy.lambdify(dimension_symbols, first_derivative) for first_derivative in first_derivatives
        ]
        self.u_second_derivatives = [
            sympy.lambdify(dimension_symbols, second_derivative) for second_derivative in second_derivatives
        ]

    def evaluate_u(self, vector):
        """
        compute the exact value of the function u for a given vector
        :param vector:
        :return: value of u and a tuple of u for each dimension
        """
        u = self.u_function(*vector)

        du_dv = (
            u_first_derivative(*vector) for u_first_derivative in self.u_first_derivatives
        )
        d2u_dv2 = (
            u_second_derivative(*vector) for u_second_derivative in self.u_second_derivatives
        )
        return u, Vector(du_dv), Vector(d2u_dv2)


if __name__ == '__main__':
    number_of_dimensions = 3
    problem = SineProblemND(number_of_dimensions)
    space = Space(4 for _ in range(number_of_dimensions))
    mesh = Mesh(space)

    for index in mesh.indices():
        point = Vector(float(index[d]) / mesh.space[d] for d in range(mesh.space.ndim))
        mesh[index] = problem.evaluate_u(point)[0]

    mesh.print("test mesh")

    if number_of_dimensions == 3:
        count = 0
        for index in mesh.indices():
            point = Vector(float(index[d]) / mesh.space[d] for d in range(mesh.space.ndim))

            a, da, d2a = problem.evaluate_u(point)
            b, db, d2b = SineProblem.evaluate_u(point)

            do_print = False
            if abs(a-b) > 1e-6:
                print("mismatch u {:12.6e} {:12.6e}".format(a, b), end="")
                do_print = True
            if not da.near(db, threshold=1e-6):
                print("mismatch du {:12} {:12}".format(da, db), end="")
                do_print = True
            if not d2a.near(d2b, threshold=1e-6):
                print("mismatch d2u {:12} {:12}".format(d2a, d2b), end="")
                do_print = True
            if do_print:
                print()

            count += 1
            if count % 10 == 0:
                print("count {}".format(count))
