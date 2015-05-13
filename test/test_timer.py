from __future__ import print_function
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.timer import Timer, EventTimer

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest


class TestTimer(unittest.TestCase):
    @staticmethod
    def big_loop():
        return [
            x*x*x for x in range(1000000)
        ]

    def test_basics(self):
        timer = EventTimer(self)

        with timer('dog'):
            TestTimer.big_loop()

        with timer('dog'):
            TestTimer.big_loop()

        with timer('cat'):
            TestTimer.big_loop()

        self.assertTrue('dog' in timer.timer_dict)
        self.assertTrue('cat' in timer.timer_dict)

        dog_timer = timer.timer_dict['dog']
        cat_timer = timer.timer_dict['cat']
        self.assertEqual(dog_timer.events, 2)
        self.assertEqual(cat_timer.events, 1)
        self.assertGreater(dog_timer.total_time, cat_timer.total_time)
        timer.show_timers()

    def test_level_timer(self):
        solver = SimpleMultigridSolver.get_solver(["3"])
        level = solver.fine_level
        self.assertIsInstance(level.timer, EventTimer)

        level.timer.clear()

        with level.timer("fox"):
            TestTimer.big_loop()

        with level.timer("emu"):
            TestTimer.big_loop()

        with level.timer("goat"):
            TestTimer.big_loop()

        print(level.timer.names())
        self.assertEqual(len(level.timer.names()), 3)
        self.assertIn("fox", level.timer.names())
        self.assertNotIn("dog", level.timer.names())

        level.timer.show_timers()


