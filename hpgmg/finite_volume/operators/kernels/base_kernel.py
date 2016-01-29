import abc

__author__ = 'nzhang-dev'


class BaseKernel(object):
    @abc.abstractmethod
    def get_stencil(self):
        pass