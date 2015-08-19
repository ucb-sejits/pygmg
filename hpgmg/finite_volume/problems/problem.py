from __future__ import print_function

from hpgmg.finite_volume.hpgmg_exception import HpgmgException
__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Problem(object):
    def evaluate_u(self, vector):
        """
        compute the exact value of the function u for a given point in space
        :param vector:
        :return: value of u and a tuple of u for each dimension
        """
        raise HpgmgException("Problem subclass must implement evaluate_u")
