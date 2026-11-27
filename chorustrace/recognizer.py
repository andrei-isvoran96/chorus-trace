"""Training, checkpoints, and transcription."""

import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .audio import FeatureConfig, features_from_wav, log_mel, read_wav
from .ctc import log_softmax, minimum_frames
from .ctc_beam import CTCPrefixBeam
from .data import ALPHABET, Example, encode_text
from .fulltrack_decode import FullTrackTranscript, words_from_frames
from .model import DEFAULT_DILATIONS, AcousticModel, Adam, validate_dilations
from .phonemes import PhonemeSupervision

_CHECKPOINT_VERSION = 1
_DILATED_CHECKPOINT_VERSION = 2
_BLANK_OFFSET_CHECKPOINT_VERSION = 3
_INPUT_DELTAS_CHECKPOINT_VERSION = 4
_BEAM_CHECKPOINT_VERSION = 5


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.003
    hidden_size: int = 48
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)
    dilations: tuple[int, ...] = DEFAULT_DILATIONS
    input_deltas: bool = False
    ctc_prior_weight: float = 0.0
    phoneme_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.hidden_size < 1 or self.seed < 0:
            raise ValueError("epochs and hidden size must be positive and seed must be nonnegative")
        object.__setattr__(self, "dilations", validate_dilations(self.dilations))
        if not isinstance(self.input_deltas, bool):
            raise ValueError("input_deltas must be a boolean")
        if not np.isfinite(self.ctc_prior_weight) or self.ctc_prior_weight < 0:
            raise ValueError("CTC prior weight must be finite and nonnegative")
        if not np.isfinite(self.phoneme_loss_weight) or self.phoneme_loss_weight < 0:
            raise ValueError("phoneme loss weight must be finite and nonnegative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")


@dataclass
class Recognizer:
    model: AcousticModel
    features: FeatureConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_song_ids: tuple[str, ...] = ()
    test_lock_sha256: str | None = None
    blank_logit_offset: float = 0.0
    beam_width: int = 1

    def __post_init__(self) -> None:
        self.blank_logit_offset = _checked_blank_offset(self.blank_logit_offset)
        self.beam_width = _checked_beam_width(self.beam_width)

    def posteriors(self, audio: str | Path) -> np.ndarray:
        """Return an unadjusted probability distribution per frame, with blank at index 0."""
        features = features_from_wav(audio, self.features)
        logits = self.model.forward((features - self.feature_mean) / self.feature_scale)
        return np.exp(log_softmax(logits))

    def transcribe(self, audio: str | Path) -> str:
        return self.transcribe_full_track(audio).text

    def transcribe_full_track(
        self, audio: str | Path, *, segment_seconds: float | None = 20.0,
    ) -> FullTrackTranscript:
        """Decode a WAV in context-padded segments using one CTC collapse for the track.

        Pass None to run a single model forward for comparison.
        """
        if segment_seconds is not None and (not math.isfinite(segment_seconds) or segment_seconds <= 0):
            raise ValueError("segment duration must be finite and positive")
        blank_offset = _checked_blank_offset(self.blank_logit_offset)
        samples, sample_rate = read_wav(audio, preserve_channels=self.features.stereo_features)
        duration_seconds = len(samples) / sample_rate
        features = log_mel(samples, sample_rate, self.features)
        frame_seconds = self.features.hop_size / self.features.sample_rate
        segment_frames = len(features)
        if segment_seconds is not None and segment_seconds < duration_seconds:
            segment_frames = max(1, round(segment_seconds / frame_seconds))

        width = _checked_beam_width(self.beam_width)
        beam = CTCPrefixBeam(width, token_top_k=6, logit_margin=8.0) if width > 1 else None
        labels = np.empty(len(features), dtype=np.intp) if beam is None else None
        context = self.model.context_frames
        for start in range(0, len(features), segment_frames):
            end = min(start + segment_frames, len(features))
            window_start = max(0, start - context)
            window_end = min(end + context, len(features))
            window = (features[window_start:window_end] - self.feature_mean) / self.feature_scale
            scores = self.model.forward(window)[start - window_start:end - window_start]
            if blank_offset:
                scores[:, 0] -= blank_offset
            if beam is None:
                labels[start:end] = scores.argmax(axis=1)
            else:
                beam.advance(scores)
        if beam is not None:
            labels = beam.best_path()

        words = words_from_frames(labels, self.features, duration_seconds)
        boundaries = tuple(frame * frame_seconds for frame in range(segment_frames, len(features), segment_frames))
        return FullTrackTranscript(" ".join(word.text for word in words), words, boundaries)

    def save(self, path: str | Path) -> None:
        blank_offset = _checked_blank_offset(self.blank_logit_offset)
        width = _checked_beam_width(self.beam_width)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": _CHECKPOINT_VERSION,
            "alphabet": ALPHABET,
            "features": asdict(self.features),
            "hidden_size": self.model.hidden_size,
            "training_song_ids": self.training_song_ids,
            "test_lock_sha256": self.test_lock_sha256,
        }
        if not self.features.stereo_features:
            del metadata["features"]["stereo_features"]
        if blank_offset:
            metadata["version"] = _BLANK_OFFSET_CHECKPOINT_VERSION
            metadata["dilations"] = self.model.dilations
            metadata["blank_logit_offset"] = blank_offset
        elif self.model.dilations != DEFAULT_DILATIONS:
            metadata["version"] = _DILATED_CHECKPOINT_VERSION
            metadata["dilations"] = self.model.dilations
        if self.model.input_deltas:
            metadata["version"] = _INPUT_DELTAS_CHECKPOINT_VERSION
            metadata["dilations"] = self.model.dilations
            metadata["input_deltas"] = True
            metadata["blank_logit_offset"] = blank_offset
        if width > 1:
            metadata.update(version=_BEAM_CHECKPOINT_VERSION, dilations=self.model.dilations,
                            input_deltas=self.model.input_deltas, blank_logit_offset=blank_offset,
                            beam_width=width)
        parameters = {f"parameter_{name}": value for name, value in self.model.parameters.items()}
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as file:
                np.savez_compressed(
                    file, metadata=json.dumps(metadata), feature_mean=self.feature_mean,
                    feature_scale=self.feature_scale, **parameters,
                )
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "Recognizer":
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            version = metadata["version"]
            versions = (
                _CHECKPOINT_VERSION, _DILATED_CHECKPOINT_VERSION,
                _BLANK_OFFSET_CHECKPOINT_VERSION, _INPUT_DELTAS_CHECKPOINT_VERSION, _BEAM_CHECKPOINT_VERSION,
            )
            if version not in versions or metadata["alphabet"] != ALPHABET:
                raise ValueError("unsupported ChorusTrace checkpoint version or alphabet")
            features = FeatureConfig(**metadata["features"])
            dilations = DEFAULT_DILATIONS if version == _CHECKPOINT_VERSION else metadata.get("dilations")
            input_deltas = metadata.get("input_deltas") if version >= _INPUT_DELTAS_CHECKPOINT_VERSION else False
            model = AcousticModel(
                features.feature_count, len(ALPHABET) + 1, metadata["hidden_size"],
                dilations=dilations, input_deltas=input_deltas,
            )
            for name, parameter in model.parameters.items():
                parameter[:] = _checkpoint_array(saved, f"parameter_{name}", parameter.shape)
            mean = _checkpoint_array(saved, "feature_mean", (features.feature_count,))
            scale = _checkpoint_array(saved, "feature_scale", (features.feature_count,))
            if np.any(scale <= 0):
                raise ValueError("invalid feature scale in checkpoint")
            blank_offset = metadata.get("blank_logit_offset") if version >= _BLANK_OFFSET_CHECKPOINT_VERSION else 0.0
            width = metadata.get("beam_width") if version == _BEAM_CHECKPOINT_VERSION else 1
        return cls(model, features, mean, scale, tuple(metadata["training_song_ids"]),
                   metadata.get("test_lock_sha256"), blank_logit_offset=blank_offset, beam_width=width)

    def check_held_out(self, examples: Iterable[Example]) -> None:
        overlap = set(self.training_song_ids).intersection(example.song_id for example in examples)
        if overlap:
            raise ValueError(f"held-out songs overlap training: {', '.join(sorted(overlap))}")


def _checked_beam_width(width: int) -> int:
    if not isinstance(width, Integral) or isinstance(width, bool) or not 1 <= width <= 64:
        raise ValueError("beam width must be an integer between 1 and 64")
    return int(width)


def _checked_blank_offset(offset: float) -> float:
    if isinstance(offset, Real) and not isinstance(offset, bool) and 0 <= offset <= sys.float_info.max:
        return float(offset)
    raise ValueError("blank logit offset must be finite and nonnegative")


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
    validation_score: Callable[[int, Recognizer], tuple[float, float] | None] | None = None,
    feature_cache_dir: str | Path | None = None,
    phonemes: PhonemeSupervision | None = None,
) -> Recognizer:
    """Fit from random weights and optionally restore the lowest validation score.

    The scorer runs after each epoch. Return None to skip an epoch, or return
    (WER, CER) to rank it. Earlier checkpoints win exact ties.

    With a feature cache directory, keep clip features in a temporary local
    file and load one clip at a time. The file closes when training exits.
    """
    config = config or TrainingConfig()
    if bool(config.phoneme_loss_weight) != (phonemes is not None):
        raise ValueError("positive phoneme loss weight and phoneme supervision must be supplied together")
    cache = nullcontext() if feature_cache_dir is None else tempfile.TemporaryFile(
        dir=feature_cache_dir, buffering=0,
    )
    with cache as feature_file:
        return _fit(examples, config, progress, validation_score, feature_file, phonemes)


def _fit(examples, config, progress, validation_score, feature_file, phonemes):
    n_features = config.features.feature_count
    clips = []
    songs = set()
    frame_count = 0
    feature_sum = np.zeros(n_features)
    for example in examples:
        features = features_from_wav(example.audio, config.features)
        targets = encode_text(example.text)
        phone_targets = phonemes.targets.get(example.clip_id) if phonemes is not None else None
        if phone_targets is not None and minimum_frames(phone_targets) > len(features):
            raise ValueError(f"{example.audio}: phoneme target needs more than {len(features)} frames")
        required_frames = minimum_frames(targets)
        if len(features) < required_frames:
            raise ValueError(
                f"{example.audio}: target requires {required_frames} frames, got {len(features)}"
            )
        frame_count += len(features)
        feature_sum += features.sum(axis=0)
        if feature_file is None:
            clips.append((features, targets, phone_targets))
        else:
            clips.append(((feature_file.tell(), len(features)), targets, phone_targets))
            features.tofile(feature_file)
        songs.add(example.song_id)
    if not clips:
        raise ValueError("training requires at least one clip")
    if phonemes is not None and not any(target is not None for _, _, target in clips):
        raise ValueError("phoneme supervision did not match any training clips")

    def load_features(stored):
        if feature_file is None:
            return stored
        offset, rows = stored
        feature_file.seek(offset)
        values = np.fromfile(feature_file, dtype=np.float64, count=rows * n_features)
        return values.reshape(rows, n_features)

    mean = feature_sum / frame_count
    variance = sum(
        np.square(load_features(stored) - mean).sum(axis=0) for stored, _, _ in clips
    ) / frame_count
    scale = np.sqrt(np.maximum(variance, 1e-8))
    if feature_file is None:
        clips = [((features - mean) / scale, targets, phones) for features, targets, phones in clips]
    model = AcousticModel(
        n_features, len(ALPHABET) + 1, config.hidden_size, config.seed,
        dilations=config.dilations, input_deltas=config.input_deltas,
        n_phoneme_classes=phonemes.classes if phonemes is not None else None,
    )
    optimizer = Adam(model.parameters, config.learning_rate)
    phone_optimizer = Adam(model.phoneme_parameters, config.learning_rate) if phonemes is not None else None
    rng = np.random.default_rng(config.seed)
    recognizer = Recognizer(model, config.features, mean, scale, tuple(sorted(songs)))
    best_score = None
    best_parameters = {}
    log_prior_offset = None
    frame_probability_sum = None
    if config.ctc_prior_weight:
        log_prior_offset = np.full(model.n_classes, -config.ctc_prior_weight * np.log(model.n_classes))
        frame_probability_sum = np.zeros(model.n_classes)
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for index in rng.permutation(len(clips)):
            stored, targets, phone_targets = clips[index]
            features = load_features(stored)
            if feature_file is not None:
                features -= mean
                features /= scale
            auxiliary = {} if phone_targets is None else {
                "phoneme_targets": phone_targets, "phoneme_loss_weight": config.phoneme_loss_weight,
            }
            loss, gradients = model.loss_and_gradients(
                features, targets, log_prior_offset=log_prior_offset, frame_probability_sum=frame_probability_sum,
                **auxiliary,
            )
            frames = len(features)
            for gradient in gradients.values():
                gradient /= frames
            optimizer.step(gradients)
            if phone_targets is not None:
                phone_optimizer.step(gradients)
            total_loss += loss / frames
        if frame_probability_sum is not None:
            # Hold the prior fixed for each epoch and update it from that epoch's raw posteriors.
            prior = np.maximum(frame_probability_sum / frame_count, 1e-8)
            log_prior_offset = config.ctc_prior_weight * np.log(prior)
            frame_probability_sum.fill(0)
        if progress is not None:
            progress(epoch, total_loss / len(clips))
        if validation_score is not None:
            score = validation_score(epoch, recognizer)
            if score is None:
                continue
            if len(score) != 2 or not all(np.isfinite(value) and value >= 0 for value in score):
                raise ValueError("validation score must contain finite nonnegative WER and CER")
            if best_score is None or score < best_score:
                best_score = score
                best_parameters = {name: value.copy() for name, value in model.parameters.items()}
    if validation_score is not None and best_score is None:
        raise ValueError("no epoch had a validation score")
    for name, value in best_parameters.items():
        model.parameters[name][:] = value
    model.phoneme_parameters.clear()
    return recognizer
