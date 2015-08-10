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

    # def __init__(self):
    #     self._c_function = lambda: 0

    def finalize(self, entry_point_name, project_node, entry_point_typesig, context, queue, kernels):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        self.context = context
        self.queue = queue
        self.kernels = kernels
        return self

    def set_kernel_args(self, args, kwargs):
        raise NotImplementedError("PyArgs need to be Ocl-Argified")
        # need to strip self and those kinds of things, down to just the meshes needed
        # then also grab the appropriate buffers
        # assumes all kernels use the same arguments:
        # for kernel in self.kernels:
        #     kernel.args = args
        #     kernel.kwargs = kwargs

    # def __call__(self, *args, **kwargs):
    #     # FIRST SET UP ANY EXTRA MESHES YOU NEED
    #     # THEN GRAB THEIR BUFFERS
    #     # THEN SET ARGS TO THE APPROPRIATE THING
    #     # THEN SET THE KERNEL ARGS
    #     args = args + tuple(self.extra_args)
    #     self.set_kernel_args(args, kwargs)
    #     for kernel in self.kernels:
    #
    #         previous_events = [arg.evt for arg in kernel.args if hasattr(arg, "evt") and arg.evt is not None]
    #         cl.clWaitForEvents(*previous_events)
    #
    #         evt = kernel.kernel(*kernel.args, **kernel.kwargs).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
    #
    #         for arg in kernel.args:
    #             if isinstance(arg, Buffer):
    #                 arg.dirty = True
    #                 arg.evt = evt
    #
    #     return self.return_value()


class KernelRunManager(object):

    def __init__(self, kernel, global_size, local_size):
        self.kernel = kernel
        self.gsize = global_size
        self.lsize = local_size
        self.args = ()
        self.kwargs = {}