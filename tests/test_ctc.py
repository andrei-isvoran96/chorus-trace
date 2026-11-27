import math
import unittest

import numpy as np

from chorustrace.ctc import ctc_loss, greedy_decode, minimum_frames


class CTCTests(unittest.TestCase):
    def test_one_letter_sums_three_paths(self):
        probabilities = np.array([[0.4, 0.6], [0.7, 0.3]])
        # Paths 0A, A0, AA contribute 0.12, 0.42, 0.18 respectively.
        loss, gradient = ctc_loss(np.log(probabilities), [1])
        self.assertAlmostEqual(loss, -math.log(0.72))
        posterior = np.array([[0.12, 0.60], [0.42, 0.30]]) / 0.72
        np.testing.assert_allclose(gradient, probabilities - posterior, atol=1e-14)

    def test_repeated_letters_require_a_blank(self):
        probabilities = np.array([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
        self.assertEqual(minimum_frames([1, 1]), 3)
        self.assertEqual(minimum_frames([1, 2]), 2)
        with self.assertRaisesRegex(ValueError, "at least 3 frames"):
            ctc_loss(np.log(probabilities[:2]), [1, 1])
        loss, gradient = ctc_loss(np.log(probabilities), [1, 1])
        self.assertAlmostEqual(loss, -math.log(0.8 * 0.7 * 0.4))
        posterior = np.array([[0, 1], [1, 0], [0, 1]])
        np.testing.assert_allclose(gradient, probabilities - posterior, atol=1e-14)

    def test_distinct_letters_need_no_blank(self):
        probabilities = np.array([[0.2, 0.7, 0.1], [0.1, 0.4, 0.5]])
        loss, _ = ctc_loss(np.log(probabilities), [1, 2])
        self.assertAlmostEqual(loss, -math.log(0.7 * 0.5))

    def test_empty_target_is_all_blank_with_custom_blank(self):
        probabilities = np.array([[0.3, 0.7], [0.1, 0.9]])
        loss, gradient = ctc_loss(np.log(probabilities), [], blank=1)
        self.assertAlmostEqual(loss, -math.log(0.7 * 0.9))
        np.testing.assert_allclose(gradient, probabilities - [0, 1], atol=1e-14)
        empty_loss, empty_gradient = ctc_loss(np.empty((0, 2)), [])
        self.assertEqual(empty_loss, 0.0)
        self.assertEqual(empty_gradient.shape, (0, 2))

    def test_logit_gradient_matches_centered_finite_differences(self):
        logits = np.array([[0.1, -0.5, 0.3], [1.5, 1.0, 0.8], [0.6, -0.2, 0.2]])
        _, gradient = ctc_loss(logits, [1, 2])
        step = 1e-6
        for index in np.ndindex(logits.shape):
            with self.subTest(index=index):
                plus, minus = logits.copy(), logits.copy()
                plus[index] += step
                minus[index] -= step
                loss_plus, _ = ctc_loss(plus, [1, 2])
                loss_minus, _ = ctc_loss(minus, [1, 2])
                self.assertAlmostEqual(gradient[index], (loss_plus - loss_minus) / (2 * step), places=7)

    def test_invalid_target_ids(self):
        for target in ([0], [-1], [2], [1.5]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                ctc_loss(np.zeros((2, 2)), target)

    def test_greedy_merges_repeats_before_removing_blanks(self):
        scores = np.eye(3)[[0, 1, 1, 0, 1, 2, 2, 0]]
        self.assertEqual(greedy_decode(scores), [1, 1, 2])
        self.assertEqual(greedy_decode(np.eye(2)[[0, 0]]), [])
        self.assertEqual(greedy_decode(np.empty((0, 3))), [])
