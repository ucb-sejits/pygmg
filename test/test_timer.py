from __future__ import print_function
from hpgmg.finite_volume.timer import Timer

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest


class TestTimer(unittest.TestCase):
    def test_basics(self):
        with Timer('dog'):
            a = 0
            for i in range(3000000):
                a += i

        with Timer('dog'):
            a = 0
            for i in range(3000000):
                a += i

        with Timer('cat'):
            a = 0
            for i in range(100000):
                a += i

        self.assertTrue('dog' in Timer.timer_dict)
        self.assertTrue('cat' in Timer.timer_dict)

        dog_timer = Timer.timer_dict['dog']
        cat_timer = Timer.timer_dict['cat']
        self.assertEqual(dog_timer.events, 2)
        self.assertEqual(cat_timer.events, 1)
        self.assertGreater(dog_timer.total_time, cat_timer.total_time)
        Timer.show_timers()

