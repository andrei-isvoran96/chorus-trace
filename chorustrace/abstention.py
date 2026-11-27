"""Confidence gating for the greedy CTC decoder."""

import json
from pathlib import Path

import numpy as np

from .ctc import greedy_decode
from .data import CHARACTER_IDS, decode_characters, file_sha256

CALIBRATION_VERSION = 1
CONFIDENCE_METHOD = "mean_ctc_character_peak"


def decode_with_confidence(posteriors: np.ndarray) -> tuple[str, float]:
    """Return greedy text and the mean peak posterior of its non-space emissions."""
    probabilities = np.asarray(posteriors, dtype=np.float64)
    text = decode_characters(greedy_decode(probabilities))
    if not text:
        return "", 0.0
    labels = probabilities.argmax(axis=1)
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    peaks = np.maximum.reduceat(probabilities[np.arange(len(labels)), labels], starts)
    emissions = labels[starts]
    characters = (emissions != 0) & (emissions != CHARACTER_IDS[" "])
    return text, float(peaks[characters].mean())


def _validate_threshold(threshold: float) -> float:
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1):
        raise ValueError("abstention threshold must be between 0 and 1")
    return float(threshold)


def abstain(text: str, confidence: float, threshold: float) -> str:
    """Keep text whose confidence strictly exceeds the calibrated threshold."""
    return text if confidence > _validate_threshold(threshold) else ""


def load_threshold(path: str | Path, checkpoint: str | Path, *, manifest_sha256: str | None = None) -> float:
    """Read a confidence threshold bound to the exact checkpoint used to fit it."""
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(calibration, dict) or calibration.get("version") != CALIBRATION_VERSION
            or calibration.get("confidence") != CONFIDENCE_METHOD or calibration.get("split") != "validation"):
        raise ValueError("unsupported abstention calibration")
    threshold = _validate_threshold(calibration.get("threshold"))
    if calibration.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("abstention calibration belongs to a different checkpoint")
    if manifest_sha256 is not None and calibration.get("manifest_sha256") != manifest_sha256:
        raise ValueError("abstention calibration belongs to a different manifest")
    return threshold
