from __future__ import print_function
import time

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TimerRecord(object):
    def __init__(self, name):
        self.name = name
        self.total_time = 0.0
        self.events = 0

    def __str__(self):
        average = 0.0 if self.events == 0 else self.total_time / self.events
        return "timer {:30.30s} {:7d} events, {:12.5f} secs/event, {:12.5f} secs total".format(
            self.name, self.events, average, self.total_time
        )


class Timer(object):
    def __init__(self, timer_record):
        self.timer_record = timer_record
        self.starts = []

    def __enter__(self):
        self.starts.append(time.clock())
        return self

    def __exit__(self, *args):
        self.last_interval = time.clock() - self.starts.pop()
        self.timer_record.total_time += self.last_interval
        self.timer_record.events += 1

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def clear():
        Timer.timer_dict = {}


class EventTimer(object):
    def __init__(self, parent):
        self.parent = parent
        self.timer_dict = {}

    def __call__(self, name):
        if not name in self.timer_dict:
            self.timer_dict[name] = TimerRecord(name)
        return Timer(self.timer_dict[name])

    def __getitem__(self, item):
        return self.timer_dict[item]

    def clear(self):
        self.timer_dict = {}

    def names(self):
        return self.timer_dict.keys()

    def show_timers(self):
        for key in self.timer_dict.keys():
            print("{}".format(self.timer_dict[key]))
