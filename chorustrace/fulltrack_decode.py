"""Timed greedy CTC decoding of one stitched frame sequence."""

from dataclasses import dataclass

import numpy as np

from .audio import FeatureConfig
from .data import ALPHABET, normalize_text


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class FullTrackTranscript:
    text: str
    words: tuple[TimedWord, ...]
    segment_boundaries_seconds: tuple[float, ...]


def words_from_frames(
    labels: np.ndarray, config: FeatureConfig, duration_seconds: float,
) -> tuple[TimedWord, ...]:
    """Collapse CTC runs globally and time each word using its emission windows."""
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    is_character = labels[starts] != 0
    starts, ends = starts[is_character], ends[is_character]
    characters = "".join(ALPHABET[label - 1] for label in labels[starts])

    words = []
    offset = 0
    for word in normalize_text(characters).split():
        # Normalization only removes punctuation and spaces from the decoded alphabet.
        first = characters.index(word, offset)
        offset = first + len(word)
        start = int(starts[first]) * config.hop_size / config.sample_rate
        last_sample = (int(ends[offset - 1]) - 1) * config.hop_size + config.window_size
        end = last_sample / config.sample_rate
        words.append(TimedWord(word, min(start, duration_seconds), min(end, duration_seconds)))
    return tuple(words)
