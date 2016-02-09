from __future__ import print_function
from stencil_code.halo_enumerator import HaloEnumerator
from stencil_code.ordered_halo_enumerator import OrderedHaloEnumerator
from hpgmg.finite_volume.mesh import Mesh
from hpgmg.finite_volume.space import Coord

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class BoundaryUpdaterV2(object):
    """
    For cell-centered, we need to fill in the ghost zones to apply any BC's
    This code does a simple quadratic interpolation for homogeneous dirichlet (0 on boundary)
    This is first performed across faces, then to edges, then to corners (or the corresponding
    n-dimensional equivalents)

    define a continuous u(x) = ax^2 + bx + c
    For dirichlet BC's, u(0) = 0.  Thus c=0. and u(x) = ax^2+bx
    The average value of u(x) = <u_0> is expressed in terms of the integral(u(x)) = ax^3/3 + bx^2/2

    <u_0> = u[0] = a/3 + b/2 - 0
    <u_1> = u[1] = 8a/3 + 4b/2 - a/3 - b/2

    You have two equations and two unknowns (a and b).  Note, you know u[0] and u[1].
    Now, you need to find <u_-1> = 0 - (-a/3 + b/2) = a/3 - b/2

    You get <u_-1> = -5/2<u_0> + 1/2<u_1> or u[-1] = -5/2*u[0] + 1/2*u[1].
    -Sam Williams, 2015
    """
    def __init__(self, solver):
        self.solver = solver

        if self.solver.boundary_is_dirichlet:
            self.apply = BoundaryUpdaterV2.apply_dirichlet
            self.name = "dirichlet"
        elif self.solver.boundary_is_periodic:
            self.apply = BoundaryUpdaterV2.apply_periodic
            self.name = "periodic"

    @staticmethod
    def apply_dirichlet(level, mesh):
        assert(isinstance(mesh, Mesh))

        halo_iterator = OrderedHaloEnumerator(level.ghost_zone, mesh.space)

        with level.timer('apply_boundary'):
            for index, neighbor_vector in halo_iterator.point_and_neighbor_vector_iterator():
                n1 = Coord(index) + neighbor_vector
                n2 = Coord(index) + (Coord(neighbor_vector) * 2)
                mesh[index] = -2.5 * mesh[n1] + 0.5 * mesh[n2]

    @staticmethod
    def apply_periodic(level, mesh):
        assert(isinstance(mesh, Mesh))

        halo_iterator = HaloEnumerator(level.ghost_zone, mesh.space)

        def get_scale_and_neighbor(point):
            neighbor = []
            for dim in range(mesh.ndim):
                x = point[dim]
                if x < level.ghost_zone[dim]:
                    neighbor.append(mesh.space[dim]-x-2)
                elif x >= mesh.space[dim]-level.ghost_zone[dim]:
                    neighbor.append(level.ghost_zone[dim] - (x - mesh.space[dim] + 1))
                else:
                    neighbor.append(x)
            return tuple(neighbor)

        with level.timer('apply_boundary'):
            for index in halo_iterator.fixed_surface_iterator():
                neighbor_index = get_scale_and_neighbor(index)
                mesh[index] = mesh[neighbor_index]


import sympy
import numpy as np


class QuadraticBoundaryApproximation(object):
    @staticmethod
    def compute_coefficients():
        a, b, c, x = sympy.symbols('a, b, c, x')
        print("symbol(a) {} symbol(b) {}".format(a, b))
        l = []
        u = [0, 1]  # these are the givens
        # problem: how can we define "n" symbols with sympy

        # append n simultaneous equations coefficients u[i] into list
        for i in range(0,2):
            f = sympy.Poly(a*(x**2) + b*x, x)
            print("eq_{} {}".format(i+1, f))
            l.append(sympy.integrate(f, (x, i, i+1)).coeffs())

        A = np.array(l)  # load coefficients into matrix
        coefficients = np.linalg.solve(A, u)  # determine values of unknown coefficients in polynomial(a,b,c etc)
        u_on_boundary = sympy.integrate(f, (x, -1, 0)).coeffs()
        boundary_val = float(np.dot(u_on_boundary, coefficients))
        predicted = -5.0/2*u[0] + 1.0/2*u[1]

        print(A)
        print(coefficients)
        print("boundary value: {} {}".format(boundary_val, type(boundary_val)))
        print("predicted value {} {}".format(predicted, type(predicted)))
        print("{}".format(boundary_val-predicted))
        print(boundary_val == predicted)  # just a check we get the same thing as sam




if __name__ == '__main__':
    QuadraticBoundaryApproximation.compute_coefficients()

