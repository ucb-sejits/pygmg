from __future__ import print_function
from abc import ABCMeta, abstractmethod

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Smoother(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def smooth(self, level, mesh_to_smooth, rhs_mesh):
        pass