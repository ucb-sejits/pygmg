from collections import defaultdict
from ctree.tune import TuningDriver, Result
import itertools
import hpgmg

import numpy as np

__author__ = 'nzhang-dev'


class SmoothTuningDriver(TuningDriver):
    def __init__(self, objective):
        super(SmoothTuningDriver, self).__init__()
        self.iterators = {}
        self.best_results = defaultdict(Result)
        self.last_configs = {}
        self.best_configs = defaultdict(tuple)
        self.exhausted = defaultdict(bool)
        self.objective = objective
        self.args = None
        self.subconfig = None

    def _get_configs(self):
        if not hpgmg.finite_volume.CONFIG.tune:
            while True:
                #print hpgmg.finite_volume.CONFIG.block_hierarchy
                yield hpgmg.finite_volume.CONFIG.block_hierarchy
        self.args, self.subconfig = yield ()  # will always try this
        while True:
            #print(self.args, self.subconfig, self.iterators.keys())
            shape = self.subconfig['level'].interior_space
            #print(shape)
            if shape not in self.iterators:
                #print(shape)
                logs = tuple(int(np.log2(i)) for i in shape)
                iteration_space = [
                    (2**k for k in range(i+1)) for i in logs
                ]
                self.iterators[shape] = itertools.product(*iteration_space)
            #print(self.iterators[shape])
            try:
                result = next(self.iterators[shape])
                self.last_configs[shape] = result
            except StopIteration:
                result = self.best_configs[shape]
                self.exhausted[shape] = True
            response = yield result
            if response is not None:
                self.args, self.subconfig = response

    def is_exhausted(self):
        try:
            shape = self.subconfig['level'].interior_space
            return self.exhausted[shape]
        except TypeError:
            return False

    def report(self, *args, **kwargs):
        if self.subconfig is not None:
            shape = self.subconfig['level'].interior_space
            result = Result(*args, **kwargs)
            if self.objective.compare(result, self.best_results[shape]):
                self.best_results[shape] = result
                self.best_configs[shape] = self.last_configs[shape]
            print(self.best_configs[shape])