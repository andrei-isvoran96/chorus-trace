"""A small temporal acoustic model with explicit NumPy backpropagation."""

from numbers import Integral

import numpy as np

from .ctc import ctc_loss

DEFAULT_DILATIONS = (1, 3, 9)
_RESIDUAL_SCALE = 0.5
_DELTA_CONTEXT = 2


def validate_dilations(dilations):
    """Return an immutable layer order with distinct positive frame offsets."""
    try:
        values = tuple(dilations)
    except TypeError as exc:
        raise ValueError("dilations must be a sequence of distinct positive integers") from exc
    if (
        not values
        or any(not isinstance(value, Integral) or isinstance(value, bool) or value <= 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("dilations must be a sequence of distinct positive integers")
    return tuple(int(value) for value in values)


class AcousticModel:
    """Map a sequence of log-mel frames to one vector of CTC logits per frame."""

    def __init__(
        self, n_features, n_classes, hidden_size=48, seed=0, *,
        dilations=DEFAULT_DILATIONS, input_deltas=False, n_phoneme_classes=None,
    ):
        if n_features < 1 or n_classes < 2 or hidden_size < 1:
            raise ValueError("expected positive feature and hidden sizes and at least two classes")
        if not isinstance(input_deltas, bool):
            raise ValueError("input_deltas must be a boolean")
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self._dilations = validate_dilations(dilations)
        self._input_deltas = input_deltas
        self.parameters: dict[str, np.ndarray] = {}
        self.phoneme_parameters: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)

        layer_sizes = [("input", n_features, hidden_size)]
        layer_sizes += [
            (f"context_{dilation}", 3 * hidden_size, hidden_size)
            for dilation in self.dilations
        ]
        layer_sizes.append(("output", hidden_size, n_classes))
        for name, fan_in, fan_out in layer_sizes:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.parameters[f"{name}_weight"] = rng.uniform(-limit, limit, (fan_in, fan_out))
            self.parameters[f"{name}_bias"] = np.zeros(fan_out)
        if self.input_deltas:
            # Keep the same initial static encoder while letting training learn the delta contribution.
            self.parameters["delta_weight"] = np.zeros((n_features, hidden_size))
        if n_phoneme_classes is not None:
            if not isinstance(n_phoneme_classes, Integral) or n_phoneme_classes < 2:
                raise ValueError("phoneme head requires at least two classes including CTC blank")
            limit = np.sqrt(6.0 / (hidden_size + n_phoneme_classes))
            self.phoneme_parameters["phoneme_weight"] = rng.uniform(-limit, limit, (hidden_size, n_phoneme_classes))
            self.phoneme_parameters["phoneme_bias"] = np.zeros(n_phoneme_classes)

    @property
    def dilations(self) -> tuple[int, ...]:
        return self._dilations

    @property
    def input_deltas(self) -> bool:
        return self._input_deltas

    @property
    def context_frames(self) -> int:
        """Return the number of acoustic context frames on each side."""
        return sum(self.dilations) + (_DELTA_CONTEXT if self.input_deltas else 0)

    def forward(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[0]

    def _forward(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features or len(features) == 0:
            raise ValueError(f"expected nonempty frames with {self.n_features} features each")

        params = self.parameters
        projection = features @ params["input_weight"] + params["input_bias"]
        deltas = None
        if self.input_deltas:
            padded = np.pad(features, ((_DELTA_CONTEXT, _DELTA_CONTEXT), (0, 0)), mode="edge")
            nearest = padded[3:-1] - padded[1:-3]
            farthest = padded[4:] - padded[:-4]
            deltas = (nearest + 2 * farthest) / 10
            projection += deltas @ params["delta_weight"]
        projected = np.tanh(projection)
        hidden = projected
        blocks = []
        width = self.hidden_size
        for dilation in self.dilations:
            name = f"context_{dilation}"
            context = np.zeros((len(hidden), 3 * width))
            context[dilation:, :width] = hidden[:-dilation]
            context[:, width:2 * width] = hidden
            context[:-dilation, 2 * width:] = hidden[dilation:]
            activation = np.tanh(context @ params[f"{name}_weight"] + params[f"{name}_bias"])
            hidden = hidden + _RESIDUAL_SCALE * activation
            blocks.append((context, activation))

        logits = hidden @ params["output_weight"] + params["output_bias"]
        return logits, features, deltas, projected, blocks, hidden

    def loss_and_gradients(
        self, features, targets, *, log_prior_offset=None, frame_probability_sum=None,
        phoneme_targets=None, phoneme_loss_weight=0.0,
    ):
        """Backpropagate character CTC and optional training-only phoneme CTC."""
        if phoneme_loss_weight and (not np.isfinite(phoneme_loss_weight) or phoneme_loss_weight < 0):
            raise ValueError("phoneme loss weight must be finite and nonnegative")
        use_phonemes = phoneme_loss_weight > 0 and phoneme_targets is not None
        if use_phonemes and not self.phoneme_parameters:
            raise ValueError("phoneme targets require a configured phoneme head")
        logits, features, deltas, projected, blocks, hidden = self._forward(features)
        loss, logit_gradient = ctc_loss(
            logits, targets, log_prior_offset=log_prior_offset, frame_probability_sum=frame_probability_sum,
        )
        params = self.parameters
        gradients = {
            "output_weight": hidden.T @ logit_gradient,
            "output_bias": logit_gradient.sum(axis=0),
        }
        hidden_gradient = logit_gradient @ params["output_weight"].T
        if use_phonemes:
            phoneme_weight = self.phoneme_parameters["phoneme_weight"]
            phoneme_logits = hidden @ phoneme_weight + self.phoneme_parameters["phoneme_bias"]
            phoneme_loss, phoneme_gradient = ctc_loss(phoneme_logits, phoneme_targets)
            loss += phoneme_loss_weight * phoneme_loss
            phoneme_gradient *= phoneme_loss_weight
            gradients["phoneme_weight"] = hidden.T @ phoneme_gradient
            gradients["phoneme_bias"] = phoneme_gradient.sum(axis=0)
            hidden_gradient += phoneme_gradient @ phoneme_weight.T
        width = self.hidden_size

        for dilation, (context, activation) in zip(reversed(self.dilations), reversed(blocks)):
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
        if deltas is not None:
            gradients["delta_weight"] = deltas.T @ projected_gradient
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
