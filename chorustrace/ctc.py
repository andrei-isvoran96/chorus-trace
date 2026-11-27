"""Connectionist temporal classification and greedy path decoding."""

import numpy as np


def _score_matrix(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite frames-by-classes matrix")
    return scores


def _check_blank(blank, classes):
    if not isinstance(blank, (int, np.integer)) or not 0 <= blank < classes:
        raise ValueError("blank must index a score column")


def _target_ids(targets):
    ids = np.asarray(targets)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError("targets must be a one-dimensional sequence of integer IDs")
    return ids


def minimum_frames(targets) -> int:
    """Return the shortest valid path length for a target sequence."""
    ids = _target_ids(targets)
    return ids.size + int(np.count_nonzero(ids[1:] == ids[:-1]))


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Return log probabilities for each frame."""
    scores = _score_matrix(logits)
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def ctc_loss(
    logits: np.ndarray, targets, blank: int = 0, *,
    log_prior_offset: np.ndarray | None = None,
    frame_probability_sum: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return CTC loss and its gradient with respect to logits.

    A fixed per-class log prior offset changes path weights. An optional
    accumulator receives the sum of raw softmax probabilities across frames.
    """
    log_probs = log_softmax(logits)
    frames, classes = log_probs.shape
    _check_blank(blank, classes)
    if log_prior_offset is not None:
        log_prior_offset = np.asarray(log_prior_offset, dtype=np.float64)
        if log_prior_offset.shape != (classes,) or not np.isfinite(log_prior_offset).all():
            raise ValueError("log prior offset must have one finite value per class")
    if frame_probability_sum is not None and frame_probability_sum.shape != (classes,):
        raise ValueError("frame probability accumulator must have one value per class")
    ids = _target_ids(targets)
    if np.any((ids < 0) | (ids >= classes) | (ids == blank)):
        raise ValueError("target IDs must index non-blank score columns")
    required = minimum_frames(ids)
    if frames < required:
        raise ValueError(f"CTC target needs at least {required} frames, got {frames}")
    if frames == 0:
        return 0.0, np.zeros_like(log_probs)

    labels = np.full(2 * ids.size + 1, blank, dtype=np.intp)
    labels[1::2] = ids
    skip_to = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    emissions = log_probs[:, labels]
    if log_prior_offset is not None:
        emissions -= log_prior_offset[labels]
    end_states = min(2, labels.size)

    # Alpha includes this frame's emission. A skip jumps over the blank only
    # when the two neighboring target labels differ.
    alpha = np.empty_like(emissions)
    alpha[0] = -np.inf
    alpha[0, :end_states] = emissions[0, :end_states]
    for time in range(1, frames):
        previous = alpha[time - 1]
        current = alpha[time]
        current[0] = previous[0]
        np.logaddexp(previous[1:], previous[:-1], out=current[1:])
        np.logaddexp(current[2:], previous[:-2], where=skip_to, out=current[2:])
        current += emissions[time]
    log_likelihood = float(np.logaddexp.reduce(alpha[-1, -end_states:]))

    # Beta accounts for emissions after this frame. End at the last label or
    # the trailing blank (only the blank exists for an empty target).
    beta = np.empty_like(emissions)
    beta[-1] = -np.inf
    beta[-1, -end_states:] = 0.0
    for time in range(frames - 2, -1, -1):
        next_scores = beta[time + 1] + emissions[time + 1]
        current = beta[time]
        current[-1] = next_scores[-1]
        np.logaddexp(next_scores[:-1], next_scores[1:], out=current[:-1])
        np.logaddexp(current[:-2], next_scores[2:], where=skip_to, out=current[:-2])

    # Collect alternative alignment states that emit the same class.
    posterior = np.zeros_like(log_probs)
    np.add.at(posterior.T, labels, np.exp(alpha + beta - log_likelihood).T)
    probabilities = np.exp(log_probs)
    if frame_probability_sum is not None:
        frame_probability_sum += probabilities.sum(axis=0)
    return -log_likelihood, probabilities - posterior


def greedy_decode(scores: np.ndarray, blank: int = 0) -> list[int]:
    """Take the best class per frame, merge adjacent repeats, and drop blanks."""
    values = _score_matrix(scores)
    _check_blank(blank, values.shape[1])
    decoded = []
    previous = blank
    for label in values.argmax(axis=1).tolist():
        if label != previous and label != blank:
            decoded.append(label)
        previous = label
    return decoded
