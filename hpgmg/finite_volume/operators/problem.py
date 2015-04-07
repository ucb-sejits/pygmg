from __future__ import print_function

from hpgmg.finite_volume.hpgmg_exception import HpgmgException
__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Problem(object):
    @staticmethod
    def evaluate_beta(vector):
        """
        :param vector: compute values for beta at point in vector
        :return: a beta value for center, and for each lower face about vector
        """
        raise HpgmgException("Problem subclass must implement evaluate_beta")

    @staticmethod
    def evaluate_u(coord):
        """
        compute the exact value of the function u for a given coordinate
        :param coord:
        :return: value of u and a tuple of u for each dimension
        """
        raise HpgmgException("Problem subclass must implement evaluate_beta")

    @staticmethod
    def setup(level, h_level, a, b, is_variable_coefficient):
        """
        instantiate a level with the data for this problem
        :param level:
        :param h_level:
        :param a:
        :param b:
        :param is_variable_coefficient:
        :return:
        """
        raise HpgmgException("Problem subclass must implement evaluate_beta")
