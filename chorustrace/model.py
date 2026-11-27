"""A small temporal acoustic model with explicit NumPy backpropagation."""

import numpy as np

from .ctc import ctc_loss

_DILATIONS = (1, 3, 9)
_RESIDUAL_SCALE = 0.5


class AcousticModel:
    """Map a sequence of log-mel frames to one vector of CTC logits per frame."""

    def __init__(self, n_features, n_classes, hidden_size=48, seed=0):
        if n_features < 1 or n_classes < 2 or hidden_size < 1:
            raise ValueError("expected positive feature and hidden sizes and at least two classes")
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.parameters: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)

        layer_sizes = [("input", n_features, hidden_size)]
        layer_sizes += [
            (f"context_{dilation}", 3 * hidden_size, hidden_size)
            for dilation in _DILATIONS
        ]
        layer_sizes.append(("output", hidden_size, n_classes))
        for name, fan_in, fan_out in layer_sizes:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.parameters[f"{name}_weight"] = rng.uniform(-limit, limit, (fan_in, fan_out))
            self.parameters[f"{name}_bias"] = np.zeros(fan_out)

    def forward(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[0]

    def _forward(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features or len(features) == 0:
            raise ValueError(f"expected nonempty frames with {self.n_features} features each")

        params = self.parameters
        projected = np.tanh(features @ params["input_weight"] + params["input_bias"])
        hidden = projected
        blocks = []
        width = self.hidden_size
        for dilation in _DILATIONS:
            name = f"context_{dilation}"
            context = np.zeros((len(hidden), 3 * width))
            context[dilation:, :width] = hidden[:-dilation]
            context[:, width:2 * width] = hidden
            context[:-dilation, 2 * width:] = hidden[dilation:]
            activation = np.tanh(context @ params[f"{name}_weight"] + params[f"{name}_bias"])
            hidden = hidden + _RESIDUAL_SCALE * activation
            blocks.append((context, activation))

        logits = hidden @ params["output_weight"] + params["output_bias"]
        return logits, features, projected, blocks, hidden

    def loss_and_gradients(self, features, targets):
        logits, features, projected, blocks, hidden = self._forward(features)
        loss, logit_gradient = ctc_loss(logits, targets)
        params = self.parameters
        gradients = {
            "output_weight": hidden.T @ logit_gradient,
            "output_bias": logit_gradient.sum(axis=0),
        }
        hidden_gradient = logit_gradient @ params["output_weight"].T
        width = self.hidden_size

        for dilation, (context, activation) in zip(reversed(_DILATIONS), reversed(blocks)):
            name = f"context_{dilation}"
            activation_gradient = _RESIDUAL_SCALE * hidden_gradient * (1 - activation**2)
            gradients[f"{name}_weight"] = context.T @ activation_gradient
            gradients[f"{name}_bias"] = activation_gradient.sum(axis=0)
            context_gradient = activation_gradient @ params[f"{name}_weight"].T

            # The residual path already supplies hidden_gradient. Sum the gradients
            # of every context slot that also read each frame.
            hidden_gradient[:-dilation] += context_gradient[dilation:, :width]
            hidden_gradient += context_gradient[:, width:2 * width]
            hidden_gradient[dilation:] += context_gradient[:-dilation, 2 * width:]

        projected_gradient = hidden_gradient * (1 - projected**2)
        gradients["input_weight"] = features.T @ projected_gradient
        gradients["input_bias"] = projected_gradient.sum(axis=0)
        return loss, gradients


class Adam:
    """Update the model parameters in place using bias-corrected moments."""

    def __init__(self, parameters, learning_rate=0.003):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        self.learning_rate = learning_rate
        self._parameters = parameters
        self._first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self._second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self._steps = 0

    def step(self, gradients):
        beta1 = 0.9
        beta2 = 0.999
        self._steps += 1
        first_correction = 1 - beta1**self._steps
        second_correction = 1 - beta2**self._steps

        for name, parameter in self._parameters.items():
            gradient = gradients[name]
            first = self._first[name]
            second = self._second[name]
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * gradient**2
            mean = first / first_correction
            variance = second / second_correction
            parameter -= self.learning_rate * mean / (np.sqrt(variance) + 1e-8)
