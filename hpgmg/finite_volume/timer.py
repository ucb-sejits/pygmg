from __future__ import print_function
import time

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Timer:  # pragma: no cover
    timer_dict = {}

    def __init__(self, timer_name):
        if not timer_name in Timer.timer_dict:
            Timer.timer_dict[timer_name] = 0.0
        self.timer_name = timer_name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.last_interval = time.clock() - self.start
        Timer.timer_dict[self.timer_name] += self.last_interval

    @staticmethod
    def show_timers():
        for name in sorted(Timer.timer_dict.keys()):
            print("{:30.30s} {:12.5f}".format(name, Timer.timer_dict[name]))
