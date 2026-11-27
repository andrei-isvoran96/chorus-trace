"""Score a reloaded checkpoint on one split of a verified lyric benchmark."""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from chorustrace.abstention import abstain, decode_with_confidence, load_threshold
from chorustrace.data import (
    SPLITS,
    Example,
    file_sha256,
    read_jsonl,
    read_manifest,
    string_field,
)
from chorustrace.evaluation import ErrorCounts
from chorustrace.recognizer import Recognizer

from .prepare_lyrics import verify_frozen


@dataclass(frozen=True)
class BenchmarkClip:
    example: Example
    artist: str
    genre: str
    recording: str
    duration_seconds: float


def audit_manifest(manifest: Path, audio_sha256: dict[str, str]) -> list[BenchmarkClip]:
    """Independently check split identities and matched conditions in the frozen rows."""
    artist_splits, song_splits, recording_splits, audio_splits = {}, {}, {}, {}
    recordings, pairs, conditions = {}, {}, set()
    clips = []
    for example, (location, row) in zip(read_manifest(manifest), read_jsonl(manifest), strict=True):
        artist = string_field(row, "artist", location)
        genre = string_field(row, "genre", location)
        recording = string_field(row, "recording_id", location)
        clip = string_field(row, "clip_id", location)
        condition = string_field(row, "condition", location)
        if example.split not in SPLITS:
            raise ValueError(f"{location}: an explicit split is required")
        for key, splits, label in (
            (artist.casefold(), artist_splits, "artist"),
            (example.song_id, song_splits, "song"),
            (recording, recording_splits, "recording"),
        ):
            if splits.setdefault(key, example.split) != example.split:
                raise ValueError(f"{location}: {label} {key!r} spans splits")
        if recordings.setdefault(recording, (example.song_id, artist)) != (example.song_id, artist):
            raise ValueError(f"{location}: recording {recording!r} has conflicting song or artist IDs")
        if (condition == "instrumental") != (example.text == ""):
            raise ValueError(f"{location}: instrumental clips require empty lyrics and voiced clips require lyrics")
        if example.text:
            digest = audio_sha256[row["audio"]]
            if audio_splits.setdefault(digest, example.split) != example.split:
                raise ValueError(f"{location}: duplicate voiced audio spans splits")
        key = example.song_id, clip
        metadata = artist, genre, recording, example.split, row["start"], row["end"], row["duration_seconds"]
        if pairs.setdefault(key, metadata) != metadata or (*key, condition) in conditions:
            raise ValueError(f"{location}: conflicting metadata or repeated condition for {key!r}")
        conditions.add((*key, condition))
        clips.append(BenchmarkClip(example, artist, genre, recording, row["duration_seconds"]))

    references = {
        (clip.example.song_id, clip.example.clip_id): clip.example.text
        for clip in clips if clip.example.condition in ("vocal", "original_mix")
    }
    for clip in clips:
        example = clip.example
        reference = references.get((example.song_id, example.clip_id))
        if reference is None or (example.text and example.text != reference):
            raise ValueError(f"unmatched lyric reference for {example.song_id}/{example.clip_id}")
    return clips


def verify_checkpoint(recognizer: Recognizer, clips: list[BenchmarkClip], manifest_sha256: str) -> None:
    """Confirm checkpoint provenance and both levels of split exclusion."""
    if recognizer.training_manifest_sha256 != manifest_sha256:
        raise ValueError("checkpoint is missing provenance for this frozen manifest")
    training_songs = set(recognizer.training_song_ids)
    training_rows = [clip for clip in clips if clip.example.split == "train"]
    if not training_songs or not training_songs <= {clip.example.song_id for clip in training_rows}:
        raise ValueError("checkpoint training songs must be present in the frozen training split")
    expected_artists = {clip.artist for clip in training_rows if clip.example.song_id in training_songs}
    training_artists = set(recognizer.training_artist_ids)
    if not training_artists or not training_artists <= expected_artists:
        raise ValueError("checkpoint training artist IDs disagree with the frozen training split")
    matched_songs = {
        clip.example.song_id for clip in training_rows if clip.artist in training_artists
    }
    if not training_songs <= matched_songs:
        raise ValueError("checkpoint training artist IDs do not cover its training songs")
    holdout = [clip for clip in clips if clip.example.split != "train"]
    recognizer.check_held_out(clip.example for clip in holdout)
    if training_artists & {clip.artist for clip in holdout}:
        raise ValueError("checkpoint training artists overlap held-out artists")


class Scores:
    def __init__(self):
        self.errors = ErrorCounts()
        self.greedy_errors = ErrorCounts()
        self.samples = []
        self.voiced = self.instrumentals = 0
        self.voiced_rejects = self.voiced_gate_suppressions = 0
        self.false_alarms = self.greedy_false_alarms = 0

    def add(self, result: dict, samples: int) -> None:
        reference, prediction, greedy = (result[key] for key in ("reference", "prediction", "greedy"))
        if reference:
            self.voiced += 1
            self.errors.add(reference, prediction)
            self.greedy_errors.add(reference, greedy)
            self.voiced_rejects += not prediction
            self.voiced_gate_suppressions += bool(greedy and not prediction)
        else:
            self.instrumentals += 1
            self.false_alarms += bool(prediction)
            self.greedy_false_alarms += bool(greedy)
        if len(self.samples) < samples:
            self.samples.append(result)

    def report(self) -> dict:
        return {
            "clips": self.voiced + self.instrumentals,
            "voiced_clips": self.voiced,
            "instrumental_clips": self.instrumentals,
            "instrumental_with_words": self.false_alarms,
            "instrumental_word_rate": self.false_alarms / self.instrumentals if self.instrumentals else None,
            "voiced_rejects": self.voiced_rejects,
            "voiced_reject_rate": self.voiced_rejects / self.voiced if self.voiced else None,
            "voiced_gate_suppressions": self.voiced_gate_suppressions,
            **self.errors.report(),
            "greedy": {"instrumental_with_words": self.greedy_false_alarms, **self.greedy_errors.report()},
            "samples": self.samples,
        }


def score(recognizer: Recognizer, clips: list[BenchmarkClip], samples: int, threshold: float | None) -> dict:
    if not clips or samples < 0:
        raise ValueError("scoring requires a nonempty split and a nonnegative sample count")
    conditions = defaultdict(Scores)
    genres = defaultdict(lambda: defaultdict(Scores))
    false_alarms, greedy_false_alarms = [], []
    for clip in clips:
        example = clip.example
        greedy, confidence = decode_with_confidence(recognizer.posteriors(example.audio))
        prediction = greedy if threshold is None else abstain(greedy, confidence, threshold)
        result = {
            "song_id": example.song_id,
            "clip_id": example.clip_id,
            "artist": clip.artist,
            "genre": clip.genre,
            "condition": example.condition,
            "reference": example.text,
            "prediction": prediction,
            "greedy": greedy,
            "confidence": confidence,
        }
        conditions[example.condition].add(result, samples)
        genres[clip.genre][example.condition].add(result, samples)
        if not example.text:
            if prediction:
                false_alarms.append(result)
            if threshold is not None and greedy:
                greedy_false_alarms.append(result)
    report = {
        "artists": len({clip.artist for clip in clips}),
        "songs": len({clip.example.song_id for clip in clips}),
        "recordings": len({clip.recording for clip in clips}),
        "vocal_hours": sum(clip.duration_seconds for clip in clips if clip.example.condition == "vocal") / 3600,
        "threshold": threshold,
        "conditions": {name: item.report() for name, item in sorted(conditions.items())},
        "genres": {
            genre: {name: item.report() for name, item in sorted(group.items())}
            for genre, group in sorted(genres.items())
        },
        "false_alarms": false_alarms,
    }
    if threshold is not None:
        report["greedy_false_alarms"] = greedy_false_alarms
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    try:
        manifest = verify_frozen(args.prepared)
        manifest_sha = file_sha256(manifest)
        freeze = json.loads((args.prepared / "freeze.json").read_text(encoding="utf-8"))
        clips = audit_manifest(manifest, freeze["audio_sha256"])
        recognizer = Recognizer.load(args.checkpoint)
        verify_checkpoint(recognizer, clips, manifest_sha)
        threshold = None
        if args.calibration is not None:
            threshold = load_threshold(args.calibration, args.checkpoint, manifest_sha256=manifest_sha)
        selected = [clip for clip in clips if clip.example.split == args.split]
        report = {
            "split": args.split,
            "manifest_sha256": manifest_sha,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            **score(recognizer, selected, args.samples, threshold),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
