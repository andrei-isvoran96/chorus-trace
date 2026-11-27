import unittest

import numpy as np

from chorustrace.ctc import ctc_loss
from chorustrace.model import AcousticModel, Adam


class ModelTests(unittest.TestCase):
    def test_parameter_gradients_match_finite_differences(self):
        model = AcousticModel(n_features=2, n_classes=3, hidden_size=2, seed=42)
        features = np.random.default_rng(9).normal(size=(12, 2))
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
