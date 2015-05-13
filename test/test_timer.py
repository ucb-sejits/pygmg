from __future__ import print_function
from hpgmg.finite_volume.simple_hpgmg import SimpleMultigridSolver
from hpgmg.finite_volume.timer import Timer, LevelTimer

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

import unittest


class TestTimer(unittest.TestCase):
    @staticmethod
    def big_loop():
        return [
            x*x*x for x in range(1000000)
        ]

    def test_basics(self):
        Timer.clear()

        with Timer('dog'):
            TestTimer.big_loop()

        with Timer('dog'):
            TestTimer.big_loop()

        with Timer('cat'):
            TestTimer.big_loop()

        self.assertTrue('dog' in Timer.timer_dict)
        self.assertTrue('cat' in Timer.timer_dict)

        dog_timer = Timer.timer_dict['dog']
        cat_timer = Timer.timer_dict['cat']
        self.assertEqual(dog_timer.events, 2)
        self.assertEqual(cat_timer.events, 1)
        self.assertGreater(dog_timer.total_time, cat_timer.total_time)
        Timer.show_timers()

    def test_level_timer(self):
        solver = SimpleMultigridSolver.get_solver(["3"])
        level = solver.fine_level
        self.assertIsInstance(level.timer, LevelTimer)

        Timer.clear()

        with level.timer("fox"):
            TestTimer.big_loop()

        with level.timer(["emu", "goat"]):
            TestTimer.big_loop()

        self.assertEqual(len(Timer.timer_level_dict.keys()), 1)
        print(Timer.timer_level_dict[0].keys())
        self.assertEqual(len(Timer.timer_dict.keys()), 3)
        self.assertEqual(len(Timer.timer_level_dict[0].keys()), 3)
        self.assertIn("fox", Timer.timer_dict)
        self.assertIn("emu", Timer.timer_dict)
        self.assertIn("fox", Timer.timer_level_dict[0])
        self.assertIn("emu", Timer.timer_level_dict[0])
        self.assertIn("goat", Timer.timer_level_dict[0])
        self.assertNotIn("dog", Timer.timer_dict)

        Timer.show_timers()


