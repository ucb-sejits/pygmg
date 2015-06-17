from __future__ import print_function
from abc import ABCMeta, abstractmethod
from hpgmg.finite_volume.space import Space

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class BaseOperator(object):
    __metaclass__ = ABCMeta

    def apply_op(self, mesh, index, level):
        return 0.0

    @abstractmethod
    def set_scale(self, level_h):
        pass

    @abstractmethod
    def rebuild_operator(self, target_level, source_level=None):
        pass

    def rebuild_operator_black_box(self, level, a, b, colors_in_each_dim=2):
        for dim in level.dimension_range():
            if level.dimension_size()[dim] < colors_in_each_dim:
                colors_in_each_dim = level.dimension_size()[dim]

        color_space = Space((colors_in_each_dim,) * level.dimensions)
        print("  calculating D^{-1} exactly for level h=%e using %d colors...  ".format(
            level.h, colors_in_each_dim ** level.solver.dimensions
        ))

        level.zero_mesh(level.d_inverse)
        level.zero_mesh(level.l1_inverse)

        for color_index in color_space.points:
            color_vector(level, )
            for index in level.interior_points():
                a_x = self.apply_op(level.temp, index, level)
                level.d_inverse[index] += level.temp * a_x
                level.l1_inverse[index] += abs((1.0 - level.temp[index]) * a_x)