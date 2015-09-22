from __future__ import print_function
from hpgmg.finite_volume.problems.algebraic_problem import SymmetricAlgebraicProblem

from hpgmg.finite_volume.problems.problem import Problem
from hpgmg.finite_volume.space import Vector
import sympy

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


# class ProblemP6(Problem):
#     """
#     should be continuous in u, u', u'', u''', and u'''' to guarantee high order and periodic boundaries
#     v(w) = ???
#     u(x,y,z) = v(x)v(y)v(z)
#     If Periodic, then the integral of the RHS should sum to zero.
#       Setting shift=1.0 should ensure that the integrals of X, Y, or Z should sum to zero...
#       That should(?) make the integrals of u,ux,uy,uz,uxx,uyy,uzz sum to zero and thus make the integral of f sum to zero
#     If dirichlet, then w(0)=w(1) = 0.0
#       Setting shift to 0 should ensure that U(x,y,z) = 0 on boundary
#        u =    ax^6 +    bx^5 +   cx^4 +  dx^3 +  ex^2 + fx + g
#       ux =   6ax^5 +   5bx^4 +  4cx^3 + 3dx^2 + 2ex   + f
#      uxx =  30ax^4 +  20bx^3 + 12cx^2 + 6dx   + 2e
#     a =   42.0
#     b = -126.0
#     c =  105.0
#     d =    0.0
#     e =  -21.0
#     f =    0.0
#     g =    1.0
#     """
#     def __init__(self, dimensions=3, shift=0.0):
#         self.source = []
#         expr = None
#         dimension_names = ['x', 'y', 'z'][:dimensions]
#         if dimensions > len(dimension_names):
#             dimension_names += [chr(x+ord('a')) for x in range(dimensions-len(dimension_names))]
#
#         dimension_symbols = []
#         for dimension_name in dimension_names:
#             declaration = "{0} = sympy.Symbol('{0}')".format(dimension_name)
#             self.source.append(declaration)
#             exec declaration
#             dimension_symbols.append(sympy.Symbol(dimension_name))
#
#         first_terms = [
#             "(2.0 * {sym}**6 - 6.0 * {sym}**5 + 5.0 * {sym}**4 - 1.0 * {sym}**2  + {shift})".format(sym=sym, shift=shift)
#             for sym in dimension_symbols
#         ]
#         summed_terms = " * ".join(first_terms)
#
#         text_expression = "expr = " + summed_terms
#         self.source.append(text_expression)
#
#         exec text_expression
#         # print("expression {}".format(expr))
#
#         first_derivatives = []
#         second_derivatives = []
#         for dimension_symbol in dimension_symbols:
#             first_derivatives.append(
#                 sympy.diff(expr, dimension_symbol)
#             )
#             self.source.append("du_d{}_expr = {}".format(dimension_symbol, first_derivatives[-1]))
#
#         for dimension_symbol in dimension_symbols:
#             second_derivatives.append(
#                 sympy.diff(expr, dimension_symbol, 2)
#             )
#             self.source.append("d2u_d{}2_expr = {}".format(dimension_symbol, second_derivatives[-1]))
#
#         self.u_function = sympy.lambdify(dimension_symbols, expr)
#         self.u_first_derivatives = [
#             sympy.lambdify(dimension_symbols, first_derivative) for first_derivative in first_derivatives
#         ]
#         self.u_second_derivatives = [
#             sympy.lambdify(dimension_symbols, second_derivative) for second_derivative in second_derivatives
#         ]
#
#     def evaluate_u(self, vector):
#         """
#         compute the exact value of the function u for a given vector
#         :param vector:
#         :return: value of u and a tuple of u for each dimension
#         """
#         u = self.u_function(*vector)
#
#         du_dv = (
#             u_first_derivative(*vector) for u_first_derivative in self.u_first_derivatives
#         )
#         d2u_dv2 = (
#             u_second_derivative(*vector) for u_second_derivative in self.u_second_derivatives
#         )
#         return u, Vector(du_dv), Vector(d2u_dv2)


class ProblemP6(SymmetricAlgebraicProblem):
    def __init__(self, dimensions=3, shift=0.0):
        expr = "(2.0 * x**6 - 6.0 * x**5 + 5.0 * x**4 - 1.0 * x**2  + {shift})".format(shift=shift)
        super(ProblemP6, self).__init__(expr, dimensions)