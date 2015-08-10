import abc
from ctree.jit import ConcreteSpecializedFunction
import pycl as cl
from hpgmg.finite_volume.mesh import Buffer

__author__ = 'nzhang-dev'


class PyGMGConcreteSpecializedFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self.entry_point_name = entry_point_name
        self.entry_point_typesig = entry_point_typesig
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        return self

    @staticmethod
    def pyargs_to_cargs(args, kwargs):
        raise NotImplementedError("PyArgs need to be C-Argified")

    def __call__(self, *args, **kwargs):
        result = self.pyargs_to_cargs(args, kwargs)
        cargs, ckwargs = result
        return self._c_function(*cargs, **ckwargs)


class PyGMGOclConcreteSpecializedFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig, context, queue, kernels):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.context = context
        self.queue = queue
        self.kernels = kernels
        return self

    def set_kernel_args(self, args, kwargs):
        raise NotImplementedError("PyArgs need to be Ocl-Argified")


class KernelRunManager(object):

    def __init__(self, kernel, global_size, local_size):
        self.kernel = kernel
        self.gsize = global_size
        self.lsize = local_size
        self.args = ()
        self.kwargs = {}