import abc

__author__ = 'nzhang-dev'


class KernelGenerator(object):
    """
    Describes a function that generates kernels according to the following rule:
    -1  (bottom of a dimension)
    0   (interior of a dimension)
    1   (upper face of a dimension)

    So -1, 0, 1 would be the x = 0, z = z_max interior set
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def make_kernel(self, boundary):
        pass
