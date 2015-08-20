import abc
from ctree.jit import ConcreteSpecializedFunction
import pycl as cl
from hpgmg.finite_volume.mesh import Buffer, Mesh
import ctypes
from hpgmg.finite_volume.operators.specializers.util import time_this, compute_largest_local_work_size

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

    def finalize(self, entry_point_name, project_node, entry_point_typesig, target_level, kernels, extra_args=None):
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        # self.context = context
        # self.queue = queue
        self.target_level = target_level
        self.context = self.target_level.context
        self.queue = self.target_level.queue
        self.kernels = kernels
        self.extra_args = extra_args
        return self

    def set_kernel_args(self, args, kwargs):
        kernel_args = []

        for arg in args:
            if isinstance(arg, Mesh):
                mesh = arg
                if mesh.fill_value is not None:
                    if mesh.buffer is None:
                        mesh.buffer = cl.clCreateBuffer(self.context, mesh.size * ctypes.sizeof(ctypes.c_double))
                    lsize = compute_largest_local_work_size(cl.clGetDeviceIDs()[-1], mesh.size)
                    evt = self.target_level.solver.fill_kernel(mesh.buffer.buffer, ctypes.c_double(mesh.fill_value)).on(self.queue, gsize=mesh.size, lsize=lsize)
                    mesh.buffer.evt = evt
                    mesh.dirty = False
                    mesh.fill_value = None
                elif mesh.dirty:
                    buffer = None if mesh.buffer is None else mesh.buffer.buffer
                    # buf, evt = cl.buffer_from_ndarray(self.queue, mesh, buf=buffer)
                    buf, evt = self.mesh_to_buffer(self.queue, mesh, buffer)
                    mesh.buffer = buf
                    mesh.buffer.evt = evt
                    mesh.dirty = False

                elif mesh.buffer is None:
                    size = mesh.size * ctypes.sizeof(ctypes.c_double)
                    mesh.buffer = cl.clCreateBuffer(self.context, size)

                kernel_args.append(mesh.buffer)

            elif isinstance(arg, (int, float)):
                kernel_args.append(arg)

        for kernel in self.kernels:
            kernel.args = kernel_args

    @time_this
    def mesh_to_buffer(self, queue, mesh, buffer):
        # print("\nCOPYING TO THE GPU with {}\n".format(self.__class__.__name__))
        buf, evt = cl.buffer_from_ndarray(queue, mesh, buffer)
        evt.wait()
        return buf, evt


class KernelRunManager(object):

    def __init__(self, kernel, global_size, local_size):
        self.kernel = kernel
        self.gsize = global_size
        self.lsize = local_size
        self.args = ()
        self.kwargs = {}