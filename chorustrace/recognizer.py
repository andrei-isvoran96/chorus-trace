"""Training, checkpoints, and the short-clip transcription API."""

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio import FeatureConfig, features_from_wav
from .ctc import greedy_decode, log_softmax, minimum_frames
from .data import ALPHABET, Example, decode_characters, encode_text
from .model import AcousticModel, Adam

_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.003
    hidden_size: int = 48
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.hidden_size < 1 or self.seed < 0:
            raise ValueError("epochs and hidden size must be positive and seed must be nonnegative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")


@dataclass
class Recognizer:
    model: AcousticModel
    features: FeatureConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_song_ids: tuple[str, ...] = ()

    def posteriors(self, audio: str | Path) -> np.ndarray:
        """Return a probability distribution per frame, with CTC blank at index 0."""
        features = features_from_wav(audio, self.features)
        logits = self.model.forward((features - self.feature_mean) / self.feature_scale)
        return np.exp(log_softmax(logits))

    def transcribe(self, audio: str | Path) -> str:
        return decode_characters(greedy_decode(self.posteriors(audio)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({
            "version": _CHECKPOINT_VERSION,
            "alphabet": ALPHABET,
            "features": asdict(self.features),
            "hidden_size": self.model.hidden_size,
            "training_song_ids": self.training_song_ids,
        })
        parameters = {f"parameter_{name}": value for name, value in self.model.parameters.items()}
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as file:
                np.savez_compressed(
                    file, metadata=metadata, feature_mean=self.feature_mean,
                    feature_scale=self.feature_scale, **parameters,
                )
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "Recognizer":
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata["version"] != _CHECKPOINT_VERSION or metadata["alphabet"] != ALPHABET:
                raise ValueError("unsupported ChorusTrace checkpoint version or alphabet")
            features = FeatureConfig(**metadata["features"])
            model = AcousticModel(features.n_mels, len(ALPHABET) + 1, metadata["hidden_size"])
            for name, parameter in model.parameters.items():
                parameter[:] = _checkpoint_array(saved, f"parameter_{name}", parameter.shape)
            mean = _checkpoint_array(saved, "feature_mean", (features.n_mels,))
            scale = _checkpoint_array(saved, "feature_scale", (features.n_mels,))
            if np.any(scale <= 0):
                raise ValueError("invalid feature scale in checkpoint")
        return cls(model, features, mean, scale, tuple(metadata["training_song_ids"]))

    def check_held_out(self, examples: Iterable[Example]) -> None:
        overlap = set(self.training_song_ids).intersection(example.song_id for example in examples)
        if overlap:
            raise ValueError(f"held-out songs overlap training: {', '.join(sorted(overlap))}")


def _checkpoint_array(saved, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = saved[name]
    if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"invalid {name} in checkpoint")
    return value


def train(
    examples: Iterable[Example],
    config: TrainingConfig | None = None,
    *,
    progress: Callable[[int, float], None] | None = None,
) -> Recognizer:
    """Fit a randomly initialized network and feature statistics to these examples."""
    config = config or TrainingConfig()
    clips = []
    songs = set()
    for example in examples:
        features = features_from_wav(example.audio, config.features)
        targets = encode_text(example.text)
        required_frames = minimum_frames(targets)
        if len(features) < required_frames:
            raise ValueError(
                f"{example.audio}: target requires {required_frames} frames, got {len(features)}"
            )
        clips.append((features, targets))
        songs.add(example.song_id)
    if not clips:
        raise ValueError("training requires at least one clip")

    frame_count = sum(len(features) for features, _ in clips)
    mean = sum(features.sum(axis=0) for features, _ in clips) / frame_count
    variance = sum(np.square(features - mean).sum(axis=0) for features, _ in clips) / frame_count
    scale = np.sqrt(np.maximum(variance, 1e-8))
    clips = [((features - mean) / scale, targets) for features, targets in clips]
    model = AcousticModel(config.features.n_mels, len(ALPHABET) + 1, config.hidden_size, config.seed)
    optimizer = Adam(model.parameters, config.learning_rate)
    rng = np.random.default_rng(config.seed)
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for index in rng.permutation(len(clips)):
            features, targets = clips[index]
            loss, gradients = model.loss_and_gradients(features, targets)
            frames = len(features)
            for gradient in gradients.values():
                gradient /= frames
            optimizer.step(gradients)
            total_loss += loss / frames
        if progress is not None:
            progress(epoch, total_loss / len(clips))

    return Recognizer(model, config.features, mean, scale, tuple(sorted(songs)))
