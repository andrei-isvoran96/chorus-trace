"""Score one prediction per complete original mix, with timing diagnostics."""

from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from .data import normalize_text
from .evaluation import edit_distance

if TYPE_CHECKING:
    from .fulltrack_data import FullTrack
    from .fulltrack_decode import FullTrackTranscript
    from .recognizer import Recognizer


_MATCH, _SUBSTITUTION, _DELETION, _INSERTION = range(4)
_EDIT_NAMES = (
    (_SUBSTITUTION, "substitutions"),
    (_DELETION, "deletions"),
    (_INSERTION, "insertions"),
)
_WORD_COUNTS = (*(name for _, name in _EDIT_NAMES), "word_edits")
_SCORE_COUNTS = (*_WORD_COUNTS, "reference_words", "character_edits", "reference_characters")
_PAIRED_COUNTS = ("unsegmented_word_edits", "segmentation_only_word_edits", "unsegmented_only_word_edits")
_PAIRED_NEAR_COUNTS = (
    "unsegmented_near_join_word_edits", "segmentation_only_near_join_edits", "unsegmented_only_near_join_edits",
)
_INSTRUMENTAL_COUNTS = ("intervals", "intervals_with_words", "seconds", "words")


@dataclass(frozen=True)
class _Edit:
    kind: int
    reference_position: int
    prediction_index: int | None


def _align_words(
    reference: list[str], prediction: list[str],
    reference_times: list[float] | None = None, prediction_times: list[float] | None = None,
) -> list[_Edit]:
    """Minimize edit count, then total midpoint distance among paired words."""
    if reference_times is None:
        reference_times = [0.0] * len(reference)
    if prediction_times is None:
        prediction_times = [0.0] * len(prediction)
    observations = tuple(zip(prediction, prediction_times, strict=True))
    expected_words = zip(reference, reference_times, strict=True)
    previous = [(column, 0.0) for column in range(len(prediction) + 1)]
    trace = [bytearray([_INSERTION]) * len(previous)]
    for row, (expected, expected_time) in enumerate(expected_words, start=1):
        current = [(row, 0.0)]
        back = bytearray([_DELETION])
        for column, (actual, actual_time) in enumerate(observations, start=1):
            different = expected != actual
            diagonal, diagonal_offset = previous[column - 1]
            deletion, deletion_offset = previous[column]
            insertion, insertion_offset = current[-1]
            cost, offset, direction = min(
                (diagonal + different, diagonal_offset + abs(expected_time - actual_time),
                 _SUBSTITUTION if different else _MATCH),
                (deletion + 1, deletion_offset, _DELETION),
                (insertion + 1, insertion_offset, _INSERTION),
            )
            current.append((cost, offset))
            back.append(direction)
        previous = current
        trace.append(back)

    edits = []
    row, column = len(reference), len(prediction)
    while row or column:
        kind = trace[row][column]
        if kind == _INSERTION:
            column -= 1
            edits.append(_Edit(kind, row, column))
        elif kind == _DELETION:
            row -= 1
            edits.append(_Edit(kind, row, None))
        else:
            row -= 1
            column -= 1
            if kind == _SUBSTITUTION:
                edits.append(_Edit(kind, row, column))
    return edits[::-1]


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _edit_counts(edits: list[_Edit]) -> dict:
    counts = Counter(edit.kind for edit in edits)
    return {name: counts[kind] for kind, name in _EDIT_NAMES} | {"word_edits": len(edits)}


def _error_rates(counts: dict) -> dict:
    return counts | {
        "wer": _rate(counts["word_edits"], counts["reference_words"]),
        "cer": _rate(counts["character_edits"], counts["reference_characters"]),
    }


def _prediction_times(result: "FullTrackTranscript", words: list[str]) -> list[float]:
    if [span.text for span in result.words] != words:
        raise ValueError("decoded word timings do not match transcript text")
    return [(span.start_seconds + span.end_seconds) / 2 for span in result.words]


def _instrumental(intervals: Sequence[tuple[float, float]], words: list[float]) -> dict:
    hits = [sum(start <= time < end for time in words) for start, end in intervals]
    return {
        "intervals": len(intervals),
        "intervals_with_words": sum(count > 0 for count in hits),
        "seconds": sum(end - start for start, end in intervals),
        "words": sum(hits),
    }


def _near_joins(edits: list[_Edit], reference_times: list[float], prediction_times: list[float],
                joins: Sequence[float], window: float) -> list[_Edit]:
    near = []
    for edit in edits:
        if edit.kind == _INSERTION:
            time = prediction_times[edit.prediction_index]
        else:
            time = reference_times[edit.reference_position]
        index = bisect_left(joins, time - window)
        if index < len(joins) and joins[index] <= time + window:
            near.append(edit)
    return near


def _edit_signatures(edits: list[_Edit], prediction: list[str]) -> Counter:
    return Counter(
        (edit.kind, edit.reference_position,
         prediction[edit.prediction_index] if edit.prediction_index is not None else None)
        for edit in edits
    )


def _total_fields(records: list[dict], fields: Sequence[str]) -> dict:
    return {name: sum(row[name] for row in records) for name in fields}


def _instrumental_rates(counts: dict) -> dict:
    return counts | {
        "interval_false_alarm_rate": _rate(counts["intervals_with_words"], counts["intervals"]),
        "words_per_minute": _rate(60 * counts["words"], counts["seconds"]),
    }


def evaluate_full_tracks(
    recognizer: "Recognizer",
    tracks: Sequence["FullTrack"],
    *,
    segment_seconds: float | None = 20.0,
    compare_unsegmented: bool = False,
    samples: int = 3,
    boundary_window_seconds: float = 0.5,
) -> dict:
    """Measure micro WER and CER (including spaces) and optional timing diagnostics.

    Word midpoints break ties between equally good alignments and locate edit
    events. Substitutions and deletions use reference times, while insertions and
    instrumental words use prediction times. Paired counts compare edit signatures
    across transcripts. Localization near a join requires reference word times.
    Auxiliary tracks contribute only to their separate instrumental diagnostics.
    """
    if not tracks:
        raise ValueError("full-track evaluation requires at least one track")
    if all(track.diagnostic_only for track in tracks):
        raise ValueError("full-track evaluation requires at least one lyric-referenced track")
    if samples < 0:
        raise ValueError("sample count must be nonnegative")
    if not isfinite(boundary_window_seconds) or boundary_window_seconds < 0:
        raise ValueError("boundary window must be finite and nonnegative")
    if compare_unsegmented and segment_seconds is None:
        raise ValueError("a paired comparison requires segmented decoding")

    rows, predictions, instrumentals, boundaries, paired = [], [], [], [], []
    diagnostics = []
    for track in tracks:
        if track.diagnostic_only and not track.instrumental_intervals:
            raise ValueError(f"{track.track_id}: diagnostic track requires instrumental intervals")
        result = recognizer.transcribe_full_track(track.audio, segment_seconds=segment_seconds)
        prediction = normalize_text(result.text)
        prediction_words = prediction.split()
        prediction_times = _prediction_times(result, prediction_words)
        track_details = {"track_id": track.track_id, "song_id": track.song_id, "genre": track.genre,
                         "duration_seconds": track.duration_seconds}
        if track.diagnostic_only:
            instrumental = _instrumental_rates(_instrumental(track.instrumental_intervals, prediction_times))
            diagnostics.append(track_details | {"instrumental": instrumental})
            continue

        reference = normalize_text(track.text)
        reference_words = reference.split()
        reference_times = [(span.start + span.end) / 2 for span in track.words] if track.words else None
        edits = _align_words(reference_words, prediction_words, reference_times,
                             prediction_times if reference_times is not None else None)
        row = _error_rates(_edit_counts(edits) | {
            "reference_words": len(reference_words),
            "reference_characters": len(reference),
            "character_edits": edit_distance(reference, prediction),
        })
        row.update(track_details, instrumental=None)

        if track.instrumental_intervals is not None:
            instrumental = _instrumental_rates(_instrumental(track.instrumental_intervals, prediction_times))
            row["instrumental"] = instrumental
            instrumentals.append(instrumental)

        joins = result.segment_boundaries_seconds
        boundary = {"joins": len(joins), "near_join": None, "paired": None}
        if reference_times is not None:
            near = _near_joins(edits, reference_times, prediction_times, joins, boundary_window_seconds)
            boundary["near_join"] = _edit_counts(near)
            boundaries.append(boundary)
        if compare_unsegmented:
            control = recognizer.transcribe_full_track(track.audio, segment_seconds=None)
            control_words = normalize_text(control.text).split()
            control_times = _prediction_times(control, control_words)
            identical = prediction_words == control_words
            # Identical transcripts have identical lexical edits even if word timings drift.
            control_edits = edits if identical else _align_words(
                reference_words, control_words, reference_times, control_times if reference_times is not None else None,
            )
            segmented_signatures = _edit_signatures(edits, prediction_words)
            control_signatures = _edit_signatures(control_edits, control_words)
            comparison = dict.fromkeys(_PAIRED_NEAR_COUNTS) | {
                "identical_transcript": identical,
                "unsegmented_word_edits": len(control_edits),
                "segmentation_only_word_edits": (segmented_signatures - control_signatures).total(),
                "unsegmented_only_word_edits": (control_signatures - segmented_signatures).total(),
            }
            if reference_times is not None:
                control_near = _near_joins(control_edits, reference_times, control_times,
                                           joins, boundary_window_seconds)
                comparison.update(
                    unsegmented_near_join_word_edits=len(control_near),
                    segmentation_only_near_join_edits=(
                        _edit_signatures(near, prediction_words) - control_signatures
                    ).total(),
                    unsegmented_only_near_join_edits=(
                        _edit_signatures(control_near, control_words) - segmented_signatures
                    ).total(),
                )
            boundary["paired"] = comparison
            paired.append(comparison)
        row["segment_boundaries"] = boundary
        rows.append(row)
        if len(predictions) < samples:
            predictions.append({"track_id": track.track_id, "reference": reference, "prediction": prediction})

    aggregate = _error_rates(_total_fields(rows, _SCORE_COUNTS))
    paired_timed = [boundary["paired"] for boundary in boundaries if boundary["paired"] is not None]
    paired_near_counts = _total_fields(paired_timed, _PAIRED_NEAR_COUNTS) if paired_timed else dict.fromkeys(
        _PAIRED_NEAR_COUNTS,
    )
    aggregate.update(
        duration_seconds=sum(row["duration_seconds"] for row in rows),
        instrumental_annotated_tracks=len(instrumentals),
        instrumental_annotated_duration_seconds=sum(
            row["duration_seconds"] for row in rows if row["instrumental"] is not None
        ),
        instrumental_tracks_with_intervals=sum(row["intervals"] > 0 for row in instrumentals),
        instrumental=_instrumental_rates(_total_fields(instrumentals, _INSTRUMENTAL_COUNTS)) if instrumentals else None,
        segment_boundaries={
            "joins": sum(row["segment_boundaries"]["joins"] for row in rows),
            "timed_tracks": len(boundaries),
            "timed_joins": sum(row["joins"] for row in boundaries),
            "near_join": _total_fields([row["near_join"] for row in boundaries], _WORD_COUNTS) if boundaries else None,
            "paired_tracks": len(paired),
            "paired_timed_tracks": len(paired_timed),
            "paired": (_total_fields(paired, _PAIRED_COUNTS) | paired_near_counts | {
                "identical_transcript_tracks": sum(row["identical_transcript"] for row in paired),
            }) if paired else None,
        },
    )
    report = {
        "track_count": len(rows),
        "aggregate": aggregate,
        "tracks": rows,
        "samples": predictions,
        "settings": {
            "segment_seconds": segment_seconds,
            "boundary_window_seconds": boundary_window_seconds,
            "compare_unsegmented": compare_unsegmented,
        },
    }
    if diagnostics:
        timed = [row["instrumental"] for row in diagnostics]
        report["instrumental_diagnostics"] = {
            "track_count": len(diagnostics),
            "duration_seconds": sum(row["duration_seconds"] for row in diagnostics),
            "instrumental": _instrumental_rates(_total_fields(timed, _INSTRUMENTAL_COUNTS)),
            "tracks": diagnostics,
        }
    return report
