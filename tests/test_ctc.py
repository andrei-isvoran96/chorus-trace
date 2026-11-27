import math
import unittest

import numpy as np

from chorustrace.ctc import ctc_loss, greedy_decode, minimum_frames
from chorustrace.model import AcousticModel


class CTCTests(unittest.TestCase):
    def test_one_letter_sums_three_paths(self):
        probabilities = np.array([[0.4, 0.6], [0.7, 0.3]])
        # Paths 0A, A0, AA contribute 0.12, 0.42, 0.18 respectively.
        loss, gradient = ctc_loss(np.log(probabilities), [1])
        self.assertAlmostEqual(loss, -math.log(0.72))
        posterior = np.array([[0.12, 0.60], [0.42, 0.30]]) / 0.72
        np.testing.assert_allclose(gradient, probabilities - posterior, atol=1e-14)

    def test_label_prior_weights_the_three_paths_and_keeps_raw_probabilities(self):
        probabilities = np.array([[0.4, 0.6], [0.7, 0.3]])
        offset = 0.3 * np.log([0.8, 0.2])
        adjusted = probabilities / np.exp(offset)
        blank_first = adjusted[0, 0] * adjusted[1, 1]
        blank_last = adjusted[0, 1] * adjusted[1, 0]
        repeated = adjusted[0, 1] * adjusted[1, 1]
        path_sum = blank_first + blank_last + repeated
        posterior = np.array([[blank_first, blank_last + repeated], [blank_last, blank_first + repeated]]) / path_sum
        frame_sum = np.zeros(2)
        loss, gradient = ctc_loss(
            np.log(probabilities), [1], log_prior_offset=offset, frame_probability_sum=frame_sum,
        )
        self.assertAlmostEqual(loss, -math.log(path_sum))
        np.testing.assert_allclose(gradient, probabilities - posterior, atol=1e-14)
        np.testing.assert_allclose(frame_sum, probabilities.sum(axis=0), atol=1e-14)

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
        step = 1e-6
        for offset in (None, 0.3 * np.log([0.7, 0.1, 0.2])):
            _, gradient = ctc_loss(logits, [1, 2], log_prior_offset=offset)
            for index in np.ndindex(logits.shape):
                with self.subTest(index=index, label_prior=offset is not None):
                    plus, minus = logits.copy(), logits.copy()
                    plus[index] += step
                    minus[index] -= step
                    loss_plus, _ = ctc_loss(plus, [1, 2], log_prior_offset=offset)
                    loss_minus, _ = ctc_loss(minus, [1, 2], log_prior_offset=offset)
                    self.assertAlmostEqual(gradient[index], (loss_plus - loss_minus) / (2 * step), places=7)

    def test_invalid_target_ids(self):
        for target in ([0], [-1], [2], [1.5]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                ctc_loss(np.zeros((2, 2)), target)

    def test_auxiliary_phoneme_ctc_backpropagates_into_the_shared_encoder(self):
        model = AcousticModel(2, 3, hidden_size=2, seed=7, dilations=(1, 3), n_phoneme_classes=4)
        baseline = AcousticModel(2, 3, hidden_size=2, seed=7, dilations=(1, 3))
        features = np.random.default_rng(9).normal(size=(8, 2))
        targets = [1, 2]
        options = {"phoneme_targets": [3, 1, 1], "phoneme_loss_weight": 0.2}
        loss, gradients = model.loss_and_gradients(features, targets, **options)
        main_loss, main_gradients = baseline.loss_and_gradients(features, targets)
        unlabeled_loss, unlabeled_gradients = model.loss_and_gradients(features, targets, phoneme_loss_weight=0.2)
        self.assertGreater(loss, main_loss)
        self.assertEqual(unlabeled_loss, main_loss)
        self.assertEqual(set(unlabeled_gradients), set(main_gradients))
        np.testing.assert_array_equal(model.forward(features), baseline.forward(features))
        np.testing.assert_array_equal(gradients["output_weight"], main_gradients["output_weight"])
        self.assertGreater(np.linalg.norm(gradients["input_weight"] - main_gradients["input_weight"]), 0)

        step = 1e-6
        for name, parameter in (model.parameters | model.phoneme_parameters).items():
            estimate = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + step
                above, _ = model.loss_and_gradients(features, targets, **options)
                parameter[index] = original - step
                below, _ = model.loss_and_gradients(features, targets, **options)
                parameter[index] = original
                estimate[index] = (above - below) / (2 * step)
            np.testing.assert_allclose(gradients[name], estimate, rtol=1e-5, atol=1e-7, err_msg=name)

    def test_greedy_merges_repeats_before_removing_blanks(self):
        scores = np.eye(3)[[0, 1, 1, 0, 1, 2, 2, 0]]
        self.assertEqual(greedy_decode(scores), [1, 1, 2])
        self.assertEqual(greedy_decode(np.eye(2)[[0, 0]]), [])
        self.assertEqual(greedy_decode(np.empty((0, 3))), [])
