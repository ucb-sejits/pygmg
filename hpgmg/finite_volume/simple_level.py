"""
implement a simple single threaded, variable coefficient gmg solver
"""
from __future__ import division, print_function
import math
import numpy as np
import itertools
from stencil_code.halo_enumerator import HaloEnumerator
from hpgmg.finite_volume.iterator import RangeIterator
from hpgmg.finite_volume.operators.kernels.von_neumann import ConstantCoefficientVonNeumannStencil, \
    VariableCoefficientVonNeumannStencil
from hpgmg.finite_volume.operators.specializers.mesh_op_specializers import MeshOpSpecializer, CFillMeshSpecializer, \
    CGeneralizedSimpleMeshOpSpecializer
from hpgmg.finite_volume.operators.specializers.util import time_this, specialized_func_dispatcher
from hpgmg.finite_volume.timer import EventTimer

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

from hpgmg.finite_volume.space import Vector, Space, Coord
from hpgmg.finite_volume.mesh import Mesh

import snowflake.nodes as snodes
from snowflake.vector import Vector

import hpgmg.finite_volume


class SimpleLevel(object):
            """
            From hpgmg defines.h, the list of matrices at each level
            C vector offset    python mesh name
            ================== ==================================== ===============================
            VECTOR_TEMP        temp                                 generic work space
            VECTOR_UTRUE       exact_solution
            VECTOR_F_MINUS_AV  residual
            VECTOR_F           right_hand_side
            VECTOR_U           cell_values
            VECTOR_ALPHA       alpha
            VECTOR_BETA_I      beta_face_values[0]
            VECTOR_BETA_J      beta_face_values[1]
            VECTOR_BETA_K      beta_face_values[2]
            VECTOR_DINV        d_inverse
            VECTOR_L1INV       l1_inverse
            VECTOR_VALID       valid

            """
            def __init__(self, solver, space, level_number=0):
                assert(isinstance(space, Space))

                print("attempting to create a {:d}^{:d} level (with {} BC) as level {:d}...".format(
                    space[0], solver.dimensions,
                    solver.boundary_updater.name,
                    level_number
                ))

                self.solver = solver
                self.interior_space = space
                self.space = space + (solver.ghost_zone * 2)

                self.configuration = solver.configuration
                self.is_variable_coefficient = solver.is_variable_coefficient
                self.level_number = level_number
                self.krylov_iterations = 0

                self.ghost_zone = solver.ghost_zone
                self.half_cell = Vector([0.5 for _ in range(self.solver.dimensions)])
                self.half_unit_vectors = [
                    Vector([0.5 if d == dim else 0 for d in range(self.solver.dimensions)])
                    for dim in range(self.solver.dimensions)
                ]

                self.cell_values = Mesh(self.space)
                self.right_hand_side = Mesh(self.space)
                self.exact_solution = Mesh(self.space) if self.level_number == 0 else None

                self.alpha = Mesh(self.space)
                self.beta_face_values = [
                    Mesh(self.space) for _ in range(self.solver.dimensions)
                ]
                self.valid = Mesh(self.space)
                self.valid.fill(1.0)
                self.d_inverse = Mesh(self.space)
                self.l1_inverse = Mesh(self.space)
                self.temp = Mesh(self.space)
                self.residual = Mesh(self.space)

                self.dominant_eigen_value_of_d_inv_a = 0.0

                self.cell_size = 1.0 / space[0]
                self.alpha_is_zero = None

                self.timer = EventTimer(self)


                kernel_class = None

                if solver.is_variable_coefficient:
                    kernel_class = VariableCoefficientVonNeumannStencil
                else:
                    kernel_class = ConstantCoefficientVonNeumannStencil
                self.kernel = kernel_class(self.solver.dimensions, solver.a, solver.b, 1.0 / (self.h ** 2))

                print("Level creation complete")

            def dimension_size(self):
                return self.space[0] - (self.ghost_zone[0]*2)

            def dimension_exponent(self):
                return int(math.log((self.space[0] - (self.ghost_zone[0]*2)), 2))

            def make_coarser_level(self):
                coarser_level = SimpleLevel(self.solver, (self.space-self.ghost_zone)//2, self.level_number+1)
                return coarser_level

            @property
            def h(self):
                return self.cell_size

            def indices(self):
                # for point in self.space.points:
                #     yield point
                return self.space.points

            def interior_points(self):
                return self.space.interior_points(self.ghost_zone)

            def beta_interpolation_points(self, axis):
                #print(axis)
                pts = self.space.beta_interior_points(self.ghost_zone, axis)
                #print(pts)
                return pts

            def valid_indices(self):
                # for index in self.indices():
                #     if self.valid[index] != 0.0:
                #         yield index
                return (index for index in self.indices() if self.valid[index] != 0.0)

            def ghost_indices(self, mesh):
                halo_enumerator = HaloEnumerator(self.solver.ghost, mesh.space)

                for halo_coord in halo_enumerator.fixed_surface_iterator():
                    yield halo_coord

            @time_this
            @specialized_func_dispatcher({
                'c': CFillMeshSpecializer,
                'omp': CFillMeshSpecializer
            })
            def fill_mesh_std(self, mesh, value):
                for index in self.indices():
                    mesh[index] = value  # if self.valid[index] > 0.0 else 0.0

            def fill_mesh(self, mesh, value):
                self._get_fill_stencil(mesh, value)(mesh)

            __fill_cache = {}
            def _get_fill_stencil(self, mesh, value):
                if (mesh.shape, mesh.dtype, value) in self.__fill_cache:
                    return self.__fill_cache[(mesh.shape, mesh.dtype, value)]
                weights = [0]
                for _ in range(mesh.ndim - 1):
                    weights = [weights]
                ret = self.__fill_cache[(mesh.shape, mesh.dtype, value)] = hpgmg.finite_volume.compiler.compile(
                    snodes.Stencil(
                    snodes.StencilComponent("filled", snodes.WeightArray(
                        weights
                    )) + value,
                    "filled",
                    ((0, 0, 1),)*mesh.ndim
                ))
                return ret

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def add_meshes_std(self, target_mesh, scale_a, mesh_a, scale_b, mesh_b):
                for index in self.interior_points():
                    target_mesh[index] = scale_a * mesh_a[index] + scale_b * mesh_b[index]

            __add_cache = {}
            def add_meshes(self, target_mesh, scale_a, mesh_a, scale_b, mesh_b):
                if (scale_a, scale_b) in self.__add_cache:
                    self.__add_cache[(scale_a, scale_b)](target_mesh, mesh_a, mesh_b)
                    return
                sten = snodes.Stencil(
                        snodes.StencilComponent(
                            "a", snodes.SparseWeightArray({Vector.zero_vector(target_mesh.ndim): scale_a})
                        ) + snodes.StencilComponent(
                            "b", snodes.SparseWeightArray({Vector.zero_vector(target_mesh.ndim): scale_b})
                        ),
                    "out",
                    ((1, -1, 1),) * target_mesh.ndim
                )
                func = self.__add_cache[(scale_a, scale_b)] = hpgmg.finite_volume.compiler.compile(sten)
                func(target_mesh, mesh_a, mesh_b)

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def multiply_meshes_(self, target_mesh, scale_factor, mesh_a, mesh_b):
                for index in self.interior_points():
                    target_mesh[index] = scale_factor * mesh_a[index] * mesh_b[index]

            __multiply_cache = {}
            def multiply_meshes(self, target_mesh, scale_factor, mesh_a, mesh_b):
                if (target_mesh.ndim, scale_factor) in self.__add_cache:
                    self.__add_cache[(target_mesh.ndim, scale_factor)](target_mesh, mesh_a, mesh_b)
                    return
                sten = snodes.Stencil(
                        snodes.StencilComponent(
                            "a", snodes.SparseWeightArray({Vector.zero_vector(target_mesh.ndim): scale_factor})
                        ) * snodes.StencilComponent(
                            "b", snodes.SparseWeightArray({Vector.zero_vector(target_mesh.ndim): 1})
                        ),
                    "out",
                    ((1, -1, 1),) * target_mesh.ndim
                )
                func = self.__add_cache[(target_mesh.ndim, scale_factor)] = hpgmg.finite_volume.compiler.compile(sten)
                func(target_mesh, mesh_a, mesh_b)

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def invert_mesh(self, target_mesh, scale_factor, mesh_to_invert):
                for index in self.interior_points():
                    target_mesh[index] = scale_factor / mesh_to_invert[index]

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def copy_mesh(self, target_mesh, source_mesh):
                for index in self.interior_points():
                    target_mesh[index] = source_mesh[index]

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def scale_mesh_std(self, target_mesh, scale_factor, source_mesh):
                for index in self.interior_points():
                    target_mesh[index] = scale_factor * source_mesh[index]

            def scale_mesh(self, target_mesh, scale_factor, source_mesh):
                self.__scale_mesh_stencil(target_mesh, scale_factor, source_mesh)(
                    target_mesh, source_mesh
                )

            __scale_mesh_cache = {}
            def __scale_mesh_stencil(self, target_mesh, scale_factor, source_mesh):
                if (scale_factor, source_mesh.ndim) in self.__scale_mesh_cache:
                    return self.__scale_mesh_cache[(scale_factor, source_mesh.ndim)]
                component = snodes.StencilComponent(
                    "input_mesh",
                    snodes.SparseWeightArray(
                        {Vector.zero_vector(target_mesh.ndim) : scale_factor}
                    )
                )
                ret = self.__scale_mesh_cache[(scale_factor, source_mesh.ndim)] = hpgmg.finite_volume.compiler.compile(
                    snodes.Stencil(
                        component, "target_mesh",
                        ((1, -1, 1),)*target_mesh.ndim
                    )
                )
                return ret


            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def shift_mesh(self, target_mesh, shift_value, source_mesh):
                for index in self.interior_points():
                    target_mesh[index] = shift_value + source_mesh[index]

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def dot_mesh(self, mesh_a, mesh_b):
                accumulator = 0.0
                for index in self.interior_points():
                    accumulator += mesh_a[index] * mesh_b[index]
                return accumulator

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer,
                'ocl': CGeneralizedSimpleMeshOpSpecializer
            })
            def norm_mesh(self, mesh):
                max_norm = 0.0
                for index in self.interior_points():
                    if abs(mesh[index]) > max_norm:
                        max_norm = abs(mesh[index])
                return max_norm

            @time_this
            def meshes_interiors_equal(self, mesh_a, mesh_b):
                return all(mesh_a[index] == mesh_b[index] for index in self.interior_points())

            def project_cell_to_face(self, cell_mesh, face_id):
                lower_neighbor = Vector(
                    -1 if d == face_id else 0 for d in range(len(self.space))
                )
                for index in self.valid_indices():
                    self.beta_face_values[index] = 0.5 * (cell_mesh[index + lower_neighbor] + cell_mesh[index])

            @time_this
            @specialized_func_dispatcher({
                'c': CGeneralizedSimpleMeshOpSpecializer,
                'omp': CGeneralizedSimpleMeshOpSpecializer
            })
            def mean_mesh(self, mesh):
                """
                compute the simple mean of interior of mesh
                :param mesh:
                :return:
                """
                accumulator = 0.0
                cell_count = 0
                for index in self.interior_points():
                    accumulator += mesh[index]
                    cell_count += 1
                return accumulator / cell_count

            def coord_to_cell_center_point(self, coord):
                """
                a coordinate in one of the level
                :param coord:
                :return:
                """
                # shifted = Vector(coord) - self.ghost_zone
                # halved = shifted + self.half_cell
                # result = halved * self.h
                # return result
                return ((Vector(coord) - self.ghost_zone) + self.half_cell) * self.h

            def coord_to_face_center_point(self, coord, face_dimension):
                """
                a coordinate in one of the level, shifted to the face center on the specified dimension
                :param coord:
                :return:
                """
                return self.coord_to_cell_center_point(coord) - (self.half_unit_vectors[face_dimension] * self.h)

            def print(self, title=None):
                """
                prints the cell values mesh
                :param title:
                :return:
                """
                self.cell_values.print(message=title)

            def boundary_iterator(self, boundary):
                bounds = []
                for side, dim, ghost in zip(boundary, self.space, self.ghost_zone):
                    if side == -1:
                        #lower
                        bounds.append((0, ghost))
                    elif side == 1:
                        #upper
                        bounds.append((dim - ghost, dim))
                    else:
                        #interior
                        bounds.append((ghost, dim-ghost))
                return RangeIterator(*bounds, map_func=Coord)
