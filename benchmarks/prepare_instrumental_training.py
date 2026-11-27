"""Reserve disjoint RWC solo diagnostics and stage blank-only training spans."""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from benchmarks.prepare_clip1_training import _file_hexdigest
from benchmarks.prepare_rwc_instrumental_development import METHOD, read_bank
from benchmarks.screen_clip1_development import _identity, _performer_ids
from chorustrace.data import read_jsonl, write_jsonl
from chorustrace.fulltrack_data import _check_splits, _ids, _load_selected, _read_lock, read_fulltrack_development
from chorustrace.fulltrack_training import _training_groups


_SPLIT_DOMAIN = b"rwc-solo-development-v1\0"
_MAX_TRAIN_SPAN_SECONDS = 6
_ACTIVITY_LICENSE = "https://creativecommons.org/licenses/by-nc/4.0/"


def _validate_annotation(source: Path, row: dict) -> dict:
    reference = json.loads((source.parent / row["reference"]).read_bytes())
    if (
        not row["full_track"] or row["condition"] != "original_mix"
        or reference.get("text") or reference.get("words")
        or reference.get("source_activity_license") != _ACTIVITY_LICENSE
        or METHOD not in reference.get("review_evidence", "")
        or not reference.get("checked_by", "").strip()
        or any(not re.fullmatch(r"[a-f0-9]{64}", reference.get(field, ""))
               for field in ("source_activity_archive_sha256", "source_activity_file_sha256"))
    ):
        raise ValueError(f"{row['track_id']}: expected a published, human-annotated RWC original-mix solo reference")
    return reference


def _training_spans(intervals: tuple[tuple[float, float], ...], rate: int) -> list[list[float]]:
    clipped = []
    limit = _MAX_TRAIN_SPAN_SECONDS * rate
    for begin, end in intervals:
        start_frame, end_frame = math.ceil(begin * rate), math.floor(end * rate)
        frames = end_frame - start_frame
        if frames < math.ceil(0.3 * rate):
            continue
        parts = (frames + limit - 1) // limit
        for index in range(parts):
            first, last = (start_frame + frames * step // parts for step in (index, index + 1))
            # Bias inward so the trainer's floor/ceil bounds stay in the published solo.
            clipped.append([math.nextafter(first / rate, math.inf), math.nextafter(last / rate, -math.inf)])
    return clipped


def _title_keys(row: dict) -> set[str]:
    keys = {_identity(row[field].rsplit(":", 1)[-1]) for field in ("song_id", "lyric_id", "recording_family_id")}
    if row.get("source_title"):
        keys.add(_identity(row["source_title"].split("(", 1)[0]))
    return keys - {""}


def _rebase(row: dict, source: Path, output: Path) -> dict:
    return row | {field: os.path.relpath(source.parent / row[field], output) for field in ("audio", "reference")}


def _partition(tracks) -> tuple[list[list], list[list]]:
    groups = _training_groups(tracks, seed=0)
    if len(groups) < 2:
        raise ValueError("RWC solo bank needs two disjoint artist/work/recording groups")
    # Group ranking depends only on canonical identifiers, before any scoring.
    groups.sort(key=lambda group: hashlib.sha256(
        _SPLIT_DOMAIN + "\0".join(sorted(track.track_id for track in group)).encode()
    ).digest())
    cut = len(groups) // 2
    return groups[:cut], groups[cut:]


def prepare(source: Path, development: Path, lock: Path, output: Path, *, groups: int = 276, seed: int = 0) -> Path:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    frozen = _read_lock(lock)
    originals = _load_selected(development, ("train", "validation"))
    development_rows = [row for _, row in read_jsonl(development) if row["split"] in ("train", "validation")]
    signatures = [("test", row) for row in frozen]
    signatures.extend((entry.track.split, entry.signature) for entry in originals)
    protected_rows = [*frozen, *development_rows]
    protected_titles = set().union(*(_title_keys(row) for row in protected_rows))
    protected_artists = {alias for row in protected_rows for name in row["artist_ids"] for alias in _performer_ids(name)}

    bank_rows = [row for _, row in read_jsonl(source)]
    bank = read_bank(source)
    bank_signatures = []
    annotations = {}
    for row, track in zip(bank_rows, bank, strict=True):
        aliases = {alias for artist in track.artist_ids for alias in _performer_ids(artist)}
        if _title_keys(row) & protected_titles or aliases & protected_artists:
            raise ValueError(f"{track.track_id}: RWC catalog identity overlaps protected development or test")
        annotations[track.track_id] = _validate_annotation(source, row)
        identifiers = _ids(row, track.track_id)
        bank_signatures.append(identifiers | {
            "artist_ids": sorted(set(identifiers["artist_ids"]) | aliases),
            "audio_sha256": row["audio_sha256"], "pcm_sha256": row["pcm_sha256"],
        })
    _check_splits([*signatures, *(("rwc-bank", signature) for signature in bank_signatures)])
    canonical = [replace(track, artist_ids=tuple(signature["artist_ids"]))
                 for track, signature in zip(bank, bank_signatures, strict=True)]
    train_bank, diagnostic_bank = _partition(canonical)
    negatives = {track.track_id for group in train_bank for track in group}
    _check_splits([
        ("train" if track.track_id in negatives else "validation", signature)
        for track, signature in zip(bank, bank_signatures, strict=True)
    ])

    music = [entry.track for entry in originals if entry.track.split == "train"]
    partitions = _training_groups(music, seed)
    if not 1 <= groups <= len(partitions):
        raise ValueError(f"music group count must be between 1 and {len(partitions)}")
    retained = {track.track_id for group in partitions[:groups] for track in group}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary)
        training_rows, validation_rows = [], []
        train_intervals, validation_intervals = [], []
        for row, track, signature in zip(bank_rows, bank, bank_signatures, strict=True):
            ready = _rebase(row, source, output) | {"artist_ids": signature["artist_ids"]}
            if track.track_id not in negatives:
                validation_rows.append(ready)
                validation_intervals.extend(track.instrumental_intervals)
                continue
            intervals = _training_spans(track.instrumental_intervals, row["source_sample_rate"])
            if not intervals:
                raise ValueError(f"{track.track_id}: no eligible checked solo spans for training")
            train_intervals.extend(intervals)
            label = annotations[track.track_id] | {
                "instrumental_intervals": intervals,
                "source_reference_sha256": row["reference_sha256"],
            }
            label_name = f"{track.track_id}.training.json"
            label_path = staged / label_name
            label_path.write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")
            ready.pop("diagnostic_only")
            training_rows.append(ready | {
                "split": "train", "condition": "instrumental", "reference": label_name,
                "reference_sha256": _file_hexdigest(label_path, hashlib.sha256()),
            })

        train_path = staged / "train-negatives.jsonl"
        validation_path = staged / "validation-diagnostics.jsonl"
        combined_path = staged / "combined.jsonl"
        write_jsonl(train_path, training_rows)
        write_jsonl(validation_path, validation_rows)
        base = [_rebase(row, development, output) for row in development_rows
                if row["split"] == "validation" or row["track_id"] in retained]
        write_jsonl(combined_path, base + training_rows)

        verified = read_fulltrack_development(combined_path, lock)
        read_bank(validation_path)
        actual = _training_groups([track for track in verified if track.split == "train"], seed)
        if len(actual) != groups + len(train_bank):
            raise ValueError("RWC negatives changed the selected CLIP1 group membership")
        summary = {
            "scope": "small blank-exposure pilot with reserved human-annotated solos",
            "partition_rule": "SHA-256 of domain and sorted group track IDs, lower half training",
            "negative_train": {"source_tracks": len(training_rows), "groups": len(train_bank),
                               "intervals": len(train_intervals), "seconds": sum(b - a for a, b in train_intervals),
                               "maximum_interval_seconds": max(b - a for a, b in train_intervals)},
            "reserved_validation": {"source_tracks": len(validation_rows), "groups": len(diagnostic_bank),
                                    "intervals": len(validation_intervals),
                                    "seconds": sum(b - a for a, b in validation_intervals)},
            "music_seed": seed, "music_groups": groups, "music_clips": len(retained),
            "music_validation_clips": sum(row["split"] == "validation" for row in base),
            "combined_train_groups": len(actual),
            "sha256": {name: _file_hexdigest(path, hashlib.sha256()) for name, path in (
                ("source_rwc_bank", source), ("source_development_manifest", development), ("frozen_test_lock", lock),
                ("train_negatives", train_path), ("reserved_validation", validation_path), ("combined", combined_path),
            )},
        }
        (staged / "verification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staged, output)
    return output / "combined.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "development", "lock", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--music-groups", type=int, default=276)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(prepare(args.source, args.development, args.lock, args.output, groups=args.music_groups, seed=args.seed))
