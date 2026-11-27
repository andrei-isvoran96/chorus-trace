import unittest

from scripts.learning_curve import _subset_sizes


class LearningCurveTests(unittest.TestCase):
    def test_selects_nested_whole_clip_prefixes(self):
        self.assertEqual(_subset_sizes([1800] * 21, [10, 1, 5]), [(1, 2), (5, 10), (10, 20)])
        self.assertEqual(_subset_sizes([1000] * 4, [1]), [(1, 4)])

    def test_uses_largest_available_subset_once(self):
        self.assertEqual(_subset_sizes([1800] * 5, [1, 5, 10]), [(1, 2), (None, 5)])
        self.assertEqual(_subset_sizes([1800] * 2, [1, 5, 10]), [(1, 2)])
        self.assertEqual(_subset_sizes([1800], [1, 5, 10]), [(None, 1)])

    def test_rejects_invalid_training_hours(self):
        for hours in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(hours=hours), self.assertRaisesRegex(ValueError, "positive and finite"):
                _subset_sizes([1800], hours)
