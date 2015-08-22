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
        # note: MeshReduceOpOclFunction overrides this method
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

    def __call__(self, *args, **kwargs):
        # return self.python_control(*args, **kwargs)
        return self.c_control(*args, **kwargs)

    # @time_this
    def python_control(self, *args, **kwargs):
        args_to_bufferize = self.get_all_args(args, kwargs)

        self.set_kernel_args(args_to_bufferize, kwargs)  # can move this function inside for time

        for kernel in self.kernels:
            kernel_args = []
            # previous_events = []
            for arg in kernel.args:
                if isinstance(arg, Buffer):
                    kernel_args.append(arg.buffer)
                    # if arg.evt:
                    #     previous_events.append(arg.evt)
                else:
                    kernel_args.append(arg)

            # cl.clWaitForEvents(*previous_events)
            run_evt = kernel.kernel(*kernel_args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
            # run_evt.wait()
            # self.run_kernel(kernel, kernel_args)

            # for arg in kernel.args:
            #     if isinstance(arg, Buffer):
            #         arg.evt = run_evt

        self.set_dirty_buffers(args_to_bufferize)

        return self.reduced_value()

    # @time_this
    def c_control(self, *args, **kwargs):
        args_to_bufferize = self.get_all_args(args, kwargs)
        bufferized_args = []
        # this is set_kernel_args moved inside
        for arg in args_to_bufferize:
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

                bufferized_args.append(mesh.buffer.buffer)

            elif isinstance(arg, (int, float)):
                bufferized_args.append(arg)

        control_args = [self.queue] + [kernel.kernel for kernel in self.kernels] + bufferized_args

        return_value = self._c_function(*control_args)

        self.set_dirty_buffers(args_to_bufferize)

        return return_value

    def get_all_args(self, args, kwargs):
        return args

    def set_dirty_buffers(self, args):
        # args are going to be Meshes not Buffers
        return

    def reduced_value(self):
        return None

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