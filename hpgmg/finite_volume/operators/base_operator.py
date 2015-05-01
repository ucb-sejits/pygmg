from __future__ import print_function
from abc import ABCMeta, abstractmethod

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class BaseOperator(object):
    __metaclass__ = ABCMeta

    def apply_op(self, mesh, index, level):
        pass

    @abstractmethod
    def set_scale(self, level_h):
        pass

    @abstractmethod
    def rebuild_operator(self, target_level, source_level=None):
        pass