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


class Timer(object):  # pragma: no cover
    timer_dict = {}
    timer_level_dict = {}

    def __init__(self, timer_name, level=None):
        if not timer_name in Timer.timer_dict:
            timer_record = TimerRecord(timer_name)
            Timer.timer_dict[timer_name] = timer_record
            if level is not None:
                if not level.level_number in Timer.timer_level_dict:
                    Timer.timer_level_dict[level.level_number] = {}
                Timer.timer_level_dict[level.level_number][timer_name] = timer_record
        self.timer_name = timer_name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.last_interval = time.clock() - self.start
        Timer.timer_dict[self.timer_name].total_time += self.last_interval
        Timer.timer_dict[self.timer_name].events += 1

    @staticmethod
    def show_timers():
        for name in sorted(Timer.timer_dict.keys()):
            print(Timer.timer_dict[name])


    @staticmethod
    def clear():
        Timer.timer_dict = {}
        Timer.timer_level_dict = {}


class LevelTimer(object):
    def __init__(self, level):
        self.level = level

    def __call__(self, name):
        return Timer(name, self.level)
