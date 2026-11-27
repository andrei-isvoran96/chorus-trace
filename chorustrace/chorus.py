"""Align and average posterior probabilities from manually marked chorus takes."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .ctc import greedy_decode
from .data import Example, decode_characters, normalize_text, read_jsonl, string_field
from .evaluation import ErrorCounts
from .recognizer import Recognizer


def _validate_posteriors(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all()
            or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0)):
        raise ValueError("posteriors must contain nonempty per-frame probability distributions")
    return values


def _align(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape[1] != second.shape[1]:
        raise ValueError("chorus takes must use the same classes")
    n, m = len(first), len(second)
    # Squared Hellinger distance plus a small relative-time alignment cost.
    costs = np.maximum(0, 1 - np.sqrt(first) @ np.sqrt(second).T)
    costs += 0.05 * np.abs(np.linspace(0, 1, n)[:, None] - np.linspace(0, 1, m)[None, :])
    cumulative = np.full((n + 1, m + 1), np.inf)
    cumulative[0, 0] = 0
    steps = np.empty((n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            best, step = cumulative[i, j], 0
            vertical = cumulative[i, j + 1] + 0.02
            horizontal = cumulative[i + 1, j] + 0.02
            if vertical < best:
                best, step = vertical, 1
            if horizontal < best:
                best, step = horizontal, 2
            cumulative[i + 1, j + 1] = costs[i, j] + best
            steps[i, j] = step

    path = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        step = steps[i, j]
        i -= step != 2
        j -= step != 1
    return np.array(path[::-1])


def align_posteriors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return monotone pairs of frame indices from the two takes."""
    return _align(_validate_posteriors(first), _validate_posteriors(second))


def _pool_on_anchor(anchor: np.ndarray, partner: np.ndarray, anchor_ids, partner_ids) -> np.ndarray:
    aligned = np.zeros_like(anchor)
    np.add.at(aligned, anchor_ids, partner[partner_ids])
    aligned /= np.bincount(anchor_ids, minlength=len(anchor))[:, None]
    return (anchor + aligned) / 2


def pool_posteriors(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool each take with the other, preserving each take's frame count."""
    first, second = _validate_posteriors(first), _validate_posteriors(second)
    first_ids, second_ids = _align(first, second).T
    return (
        _pool_on_anchor(first, second, first_ids, second_ids),
        _pool_on_anchor(second, first, second_ids, first_ids),
    )


def _score(reference: str, prediction: str) -> dict:
    counts = ErrorCounts()
    counts.add(reference, prediction)
    return {"prediction": prediction, **counts.report()}


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def evaluate_choruses(
    recognizer: Recognizer, examples: Sequence[Example], pairs: str | Path,
    condition: str = "original_mix",
) -> dict:
    """Compare independent and pooled decoding on each anchor in the marked pairs."""
    clips = {}
    for example in examples:
        if example.condition != condition or example.clip_id is None:
            continue
        key = example.song_id, example.clip_id
        if key in clips:
            raise ValueError(f"duplicate clip {key} in condition {condition!r}")
        clips[key] = example

    pair_reports = []
    totals = defaultdict(lambda: (ErrorCounts(), ErrorCounts()))
    posterior_cache = {}
    for location, pair in read_jsonl(Path(pairs)):
        song_id = string_field(pair, "song_id", location)
        ids = [string_field(pair, field, location) for field in ("clip_a", "clip_b")]
        keys = [(song_id, clip_id) for clip_id in ids]
        if ids[0] == ids[1] or any(key not in clips for key in keys):
            raise ValueError(f"{location}: expected two distinct clips of {song_id!r} in {condition!r}")
        references = [normalize_text(clips[key].text) for key in keys]
        if not all(references):
            raise ValueError(f"{location}: both chorus clips need lyrics")
        kind = "same_lyric" if references[0] == references[1] else "changed_lyric"
        changed_lyric = None
        lyric_take = None
        if "changed_lyric" in pair:
            changed_lyric = normalize_text(string_field(pair, "changed_lyric", location))
            matching_takes = [index for index, text in enumerate(references) if _contains_phrase(text, changed_lyric)]
            if not changed_lyric or len(matching_takes) != 1:
                raise ValueError(f"{location}: changed_lyric must occur in exactly one reference")
            lyric_take = matching_takes[0]

        for key in keys:
            if key not in posterior_cache:
                posterior_cache[key] = recognizer.posteriors(clips[key].audio)
        posteriors = [posterior_cache[key] for key in keys]
        pooled = pool_posteriors(*posteriors)
        anchors = []
        independent_stats, pooled_stats = totals[kind]
        for clip_id, reference, original, averaged in zip(ids, references, posteriors, pooled):
            independent = decode_characters(greedy_decode(original))
            combined = decode_characters(greedy_decode(averaged))
            independent_stats.add(reference, independent)
            pooled_stats.add(reference, combined)
            anchors.append({
                "clip_id": clip_id, "reference": reference,
                "independent": _score(reference, independent), "pooled": _score(reference, combined),
            })
        result = {"song_id": song_id, "kind": kind, "takes": anchors}
        if lyric_take is not None:
            present = [_contains_phrase(take["independent"]["prediction"], changed_lyric) for take in anchors]
            pooled_present = [_contains_phrase(take["pooled"]["prediction"], changed_lyric) for take in anchors]
            other = 1 - lyric_take
            result.update(
                changed_lyric=changed_lyric, changed_lyric_clip=ids[lyric_take],
                independent_has_changed_lyric=present[lyric_take], pooled_has_changed_lyric=pooled_present[lyric_take],
                erased_changed_lyric=present[lyric_take] and not pooled_present[lyric_take],
                introduced_in_other_take=pooled_present[other] and not present[other],
            )
        pair_reports.append(result)
    if not pair_reports:
        raise ValueError(f"{pairs}: no chorus pairs marked")

    return {
        "condition": condition,
        "pairs": pair_reports,
        "by_pair_type": {
            kind: {"independent": independent.report(), "pooled": pooled.report()}
            for kind, (independent, pooled) in totals.items()
        },
    }
