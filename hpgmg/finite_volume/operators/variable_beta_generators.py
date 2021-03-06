from __future__ import print_function

from math import tanh
import numpy as np
import functools
import operator
import sympy

from hpgmg.finite_volume.hpgmg_exception import HpgmgException
from hpgmg.finite_volume.space import Vector

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class VariableBeta(object):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.center = Vector(0.5 for _ in range(self.dimensions))

    def evaluate_beta(self, vector):
        """
        :param vector: compute values for beta at point in vector
        :return:
        """

        b_min = 1.0
        b_max = 10.0
        c2 = (b_max-b_min)/2  # coefficients to affect this transition
        c1 = (b_max+b_min)/2
        c3 = 10.0  # how sharply (B)eta transitions

        delta = vector - self.center
        distance = sum(d**2 for d in delta)**0.5

        beta_at_face = c1 + c2 * tanh(c3 * (distance - 0.25))

        # print("eb v {} dis {} del {} ndel {} bv {}".format(
        #     ",".join(map(str, vector)), distance, ",".join(map(str, delta)), ",".join(map(str, normalized_delta)),
        #     ",".join(map(str, beta_vector))
        # ))

        return beta_at_face

    def get_beta_expression(self, face_index=-1, alternate_var_name=None):
        b_min = 1.0
        b_max = 10.0
        c2 = (b_max-b_min)/2  # coefficients to affect this transition
        c1 = (b_max+b_min)/2
        c3 = 10.0  # how sharply (B)eta transitions

        #delta = vector - self.center
        #distance = sum(d**2 for d in delta)**0.5
        symbols = [
            sympy.Symbol("{var_name}{index}".format(
                var_name="x" if i != face_index else alternate_var_name, index=i)) for i in range(self.dimensions)]
        distance = sum((sym - d)**2 for sym, d in zip(symbols, self.center)) ** 0.5
        return c1 + c2 * sympy.tanh(c3 * (distance - 0.25))

    def get_beta_fv_expression(self, add_4th_order_correction=False, cell_size=None,
                               face_index=-1, alternate_var_name=None):
        if add_4th_order_correction and cell_size is None:
            raise HpgmgException("ProblemFv requires cells size")

        b = 0.25
        a = 2.0 * np.pi

        expression_1d = sympy.sympify("sin({period}*x)".format(period=a))
        free_symbols = expression_1d.free_symbols
        symbol = free_symbols.pop()

        symbols = [
            sympy.Symbol("{var_name}{index}".format(
                var_name="x" if i != face_index else alternate_var_name, index=i)) for i in range(self.dimensions)]
        expression_nd = functools.reduce(
            operator.mul,
            (expression_1d.xreplace({symbol: replacement_symbol}) for replacement_symbol in symbols)
        )
        # expression_nd = functools.reduce(
        #     operator.mul,
        #     (expression_1d.xreplace({symbol: sympy.Symbol("x{}".format(i))}) for i in range(self.dimensions))
        # )

        expression_nd *= sympy.sympify("{}".format(b))
        expression_nd += sympy.sympify("1.0")

        # print("Uncorrected expression for beta fv {}".format(expression_nd))

        if add_4th_order_correction:
            second_derivatives = []
            # symbols = [sympy.Symbol("x{}".format(d)) for d in range(self.dimensions)]
            # requires two loops so original expression is available for derivative
            for current_dim in range(self.dimensions):
                second_derivatives.append(expression_nd.diff(symbols[current_dim-1], 2))

            # print("using cell_size {}".format(cell_size))
            for current_dim in range(self.dimensions):
                expression_nd = expression_nd + sympy.sympify("{}*{}/24.0*{}".format(
                    cell_size, cell_size, second_derivatives[current_dim]))
        print("Corrected expression for beta fv {}".format(expression_nd))

        return expression_nd

    def evaluate_beta_vector(self, vector):
        b_min = 1.0
        b_max = 10.0
        c2 = (b_max-b_min)/2  # coefficients to affect this transition
        c3 = 10.0  # how sharply (B)eta transitions

        delta = vector - self.center
        distance = sum(d**2 for d in delta)**0.5
        normalized_delta = delta / distance
        beta_vector = c2 * c3 * normalized_delta * (1 - (tanh(c3 * (distance - 0.25))**2))
        return beta_vector

    def evaluate_beta_3d(self, vector):
        """
        :param vector: compute values for beta at point in vector
        :return:
        """

        assert self.dimensions == 3, "evaluate_beta_3d only works for 3d problems"

        x, y, z = vector

        b_min = 1.0
        b_max = 10.0
        c2 = (b_max-b_min)/2  # coefficients to affect this transition
        c1 = (b_max+b_min)/2
        c3 = 10.0  # how sharply (B)eta transitions
        x_center = 0.50
        y_center = 0.50
        z_center = 0.50
        # calculate distance from center of the domain (0.5, 0.5, 0.5)
        r2 = (x-x_center)**2 + (y-y_center)**2 + (z-z_center)**2
        r2x = 2.0*(x-x_center)
        r2y = 2.0*(y-y_center)
        r2z = 2.0*(z-z_center)

        r = r2**0.5
        rx = 0.5 * r2x * (r2 ** -0.5)
        ry = 0.5 * r2y * (r2 ** -0.5)
        rz = 0.5 * r2z * (r2 ** -0.5)
        # -------------------------
        beta = c1+c2*tanh(c3*(r-0.25))
        beta_x = c2*c3*rx * (1 - (tanh(c3*(r-0.25))**2))
        beta_y = c2*c3*ry * (1 - (tanh(c3*(r-0.25))**2))
        beta_z = c2*c3*rz * (1 - (tanh(c3*(r-0.25))**2))

        return beta, Vector(beta_x, beta_y, beta_z)
