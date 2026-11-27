import unittest

import numpy as np

from chorustrace.ctc import ctc_loss
from chorustrace.model import AcousticModel, Adam


class ModelTests(unittest.TestCase):
    def test_parameter_gradients_match_finite_differences(self):
        for architecture, frames in (("baseline", 12), ("long-context", 58)):
            with self.subTest(architecture=architecture):
                self.check_parameter_gradients(architecture, frames)

    def check_parameter_gradients(self, architecture, frames):
        model = AcousticModel(
            n_features=2, n_classes=3, hidden_size=2, seed=42, architecture=architecture
        )
        features = np.random.default_rng(9).normal(size=(frames, 2))
        targets = [1, 1, 2]
        _, gradients = model.loss_and_gradients(features, targets)

        epsilon = 1e-5
        for name, parameter in model.parameters.items():
            estimate = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                above, _ = ctc_loss(model.forward(features), targets)
                parameter[index] = original - epsilon
                below, _ = ctc_loss(model.forward(features), targets)
                parameter[index] = original
                estimate[index] = (above - below) / (2 * epsilon)
            np.testing.assert_allclose(
                gradients[name], estimate, rtol=1e-5, atol=1e-7, err_msg=name
            )

    def test_long_context_reaches_frames_outside_baseline_context(self):
        features = np.zeros((83, 2))
        changed = features.copy()
        changed[1, 0] = changed[81, 1] = 1
        baseline = AcousticModel(2, 3, hidden_size=4, seed=8)
        longer = AcousticModel(2, 3, hidden_size=4, seed=8, architecture="long-context")

        np.testing.assert_array_equal(baseline.forward(features)[41], baseline.forward(changed)[41])
        self.assertGreater(np.linalg.norm(longer.forward(changed)[41] - longer.forward(features)[41]), 1e-8)

    def test_adam_reduces_loss_on_a_shorter_sequence_than_the_context(self):
        model = AcousticModel(n_features=2, n_classes=3, hidden_size=4, seed=8)
        features = np.random.default_rng(7).normal(size=(4, 2))
        targets = [1, 2]
        optimizer = Adam(model.parameters)
        losses = []
        for _ in range(12):
            loss, gradients = model.loss_and_gradients(features, targets)
            losses.append(loss)
            optimizer.step(gradients)
        self.assertLess(losses[-1], losses[0])
