import itertools

__author__ = 'nzhang-dev'


class RangeIterator(object):
    def __init__(self, *ranges, **kwargs):
        self.repeat = kwargs.get('product', 1)
        self.map_func = kwargs.get('map_func', lambda x: x)
        self.ranges = ranges

    def __iter__(self):
        return (self.map_func(i) for i in itertools.product(*[range(low, high) for low, high in self.ranges], repeat=self.repeat))