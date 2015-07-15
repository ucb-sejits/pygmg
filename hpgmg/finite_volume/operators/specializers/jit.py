import abc
from ctree.jit import ConcreteSpecializedFunction

__author__ = 'nzhang-dev'


class PyGMGConcreteSpecializedFunction(ConcreteSpecializedFunction):

    def finalize(self, entry_point_name, project_node, entry_point_typesig):
        self.entry_point_name = entry_point_name
        self.entry_point_typesig = entry_point_typesig
        self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
        return self

    @abc.abstractmethod
    def pyargs_to_cargs(self, args, kwargs):
        pass

    def __call__(self, *args, **kwargs):
        result = self.pyargs_to_cargs(args, kwargs)
        cargs, ckwargs = result
        return self._c_function(*cargs, **ckwargs)