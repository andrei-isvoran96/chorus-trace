"""Bounded acoustic-only prefix search with a frame alignment for the result."""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


_IMPOSSIBLE = -math.inf


@dataclass(frozen=True, slots=True)
class _Run:
    token: int
    start: int
    end: int
    previous: "_Run | None"


@dataclass(slots=True)
class _PrefixScores:
    blank: float = _IMPOSSIBLE
    nonblank: float = _IMPOSSIBLE
    best_blank: float = _IMPOSSIBLE
    best_nonblank: float = _IMPOSSIBLE
    blank_path: _Run | None = None
    nonblank_path: _Run | None = None

    def total(self) -> float:
        return _logadd(self.blank, self.nonblank)

    def best(self) -> tuple[float, _Run | None]:
        if self.best_blank >= self.best_nonblank:
            return self.best_blank, self.blank_path
        return self.best_nonblank, self.nonblank_path


def _logadd(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b == _IMPOSSIBLE:
        return a
    return a + math.log1p(math.exp(b - a))


def _emit(
    states: dict[str, _PrefixScores], prefix: str, total: float, best: float,
    token: int, start: int, end: int, previous: _Run | None,
) -> None:
    state = states.get(prefix)
    if state is None:
        state = states[prefix] = _PrefixScores()
    state.nonblank = _logadd(state.nonblank, total)
    if best > state.best_nonblank:
        state.best_nonblank = best
        state.nonblank_path = _Run(token, start, end, previous)


class CTCPrefixBeam:
    """Advance one beam across sequential CTC logit batches, with blank at index 0."""

    def __init__(
        self, width: int, token_top_k: int | None = None, logit_margin: float | None = None,
    ) -> None:
        if not isinstance(width, Integral) or isinstance(width, bool) or width < 1:
            raise ValueError("beam width must be a positive integer")
        if token_top_k is not None and (
            not isinstance(token_top_k, Integral) or isinstance(token_top_k, bool) or token_top_k < 1
        ):
            raise ValueError("beam token limit must be a positive integer")
        if logit_margin is not None and (
            not isinstance(logit_margin, Real) or isinstance(logit_margin, bool)
            or not 0 <= logit_margin <= np.finfo(np.float64).max
        ):
            raise ValueError("beam logit margin must be finite and nonnegative")
        self._width = int(width)
        self._top_k = None if token_top_k is None else int(token_top_k)
        self._margin = None if logit_margin is None else float(logit_margin)
        # One code point per token makes long prefix keys compact.
        self._states = {"": _PrefixScores(blank=0.0, best_blank=0.0)}
        self._frames = 0
        self._class_count: int | None = None

    def advance(self, logits: np.ndarray) -> None:
        """Consume consecutive frames. Nonblank pruning preserves each prefix's last token."""
        scores = np.asarray(logits, dtype=np.float64)
        if scores.ndim != 2 or not 2 <= scores.shape[1] <= 0x110000:
            raise ValueError("expected a time-by-class logit matrix with at least two classes")
        if self._class_count is not None and self._class_count != scores.shape[1]:
            raise ValueError("beam classes must remain constant across batches")
        peaks = scores.max(axis=1, keepdims=True)
        if not np.isfinite(peaks).all() or np.isnan(scores).any():
            raise ValueError("each beam frame needs a finite score and no NaN or positive infinity")
        self._class_count = scores.shape[1]
        # Subtracting a shared frame constant preserves every prefix comparison.
        scores = scores - peaks
        top_tokens = None
        if self._top_k is not None and self._top_k < scores.shape[1] - 1:
            top_tokens = np.argsort(-scores[:, 1:], axis=1, kind="stable")[:, :self._top_k] + 1
        for index, row in enumerate(scores.tolist()):
            token_ids = range(1, len(row)) if top_tokens is None else top_tokens[index].tolist()
            candidates = [
                (token, chr(token), row[token]) for token in token_ids
                if row[token] != _IMPOSSIBLE and (self._margin is None or row[token] >= -self._margin)
            ]
            self._advance_frame(row, candidates)

    def _advance_frame(self, row: list[float], candidates: list[tuple[int, str, float]]) -> None:
        frame = self._frames
        end = frame + 1
        states: dict[str, _PrefixScores] = {}
        for prefix, source in self._states.items():
            total = source.total()
            best, path = source.best()
            if row[0] != _IMPOSSIBLE:
                blank = states.get(prefix)
                if blank is None:
                    blank = states[prefix] = _PrefixScores()
                blank.blank = total + row[0]
                blank.best_blank = best + row[0]
                blank.blank_path = path

            last = ord(prefix[-1]) if prefix else 0
            if last and row[last] != _IMPOSSIBLE:
                # A consecutive label lengthens its emission. A blank separates two copies.
                run = source.nonblank_path
                if run is not None:
                    _emit(states, prefix, source.nonblank + row[last], source.best_nonblank + row[last],
                          last, run.start, end, run.previous)
                if source.blank != _IMPOSSIBLE:
                    _emit(states, prefix + prefix[-1], source.blank + row[last], source.best_blank + row[last],
                          last, frame, end, source.blank_path)

            for token, char, score in candidates:
                if token != last:
                    _emit(states, prefix + char, total + score, best + score, token, frame, end, path)

        ranked = sorted(states, key=lambda prefix: states[prefix].total(), reverse=True)
        self._states = {prefix: states[prefix] for prefix in ranked[:self._width]}
        self._frames = end

    def best_path(self) -> np.ndarray:
        """Return the most probable frame path among paths for the highest-mass surviving prefix."""
        winner = max(self._states.values(), key=_PrefixScores.total)
        _, run = winner.best()
        labels = np.zeros(self._frames, dtype=np.intp)
        while run is not None:
            labels[run.start:run.end] = run.token
            run = run.previous
        return labels
