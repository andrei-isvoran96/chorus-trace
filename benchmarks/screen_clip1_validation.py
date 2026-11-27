"""Expand the official CLIP1 validation split against fixed training and test sets."""

import argparse
import csv
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from benchmarks.prepare_clip1_training import LICENSE_NAME, METADATA_NAME, METADATA_MEMBER, VALIDATION_METADATA_MEMBER
from benchmarks.screen_clip1_development import _catalog, _identity, _performer_ids, _screen
from chorustrace.data import normalize_text, read_jsonl, write_jsonl
from chorustrace.fulltrack_data import verify_fulltrack_test


def prepare(audit: Path, training_audit: Path, manifest: Path, lock: Path, output: Path) -> dict:
    """Select licensed VALID excerpts and keep the screened training shortlist unchanged."""
    training = [row for _, row in read_jsonl(training_audit / "shortlist.jsonl")]
    if not training:
        raise ValueError(f"{training_audit}: a fixed training shortlist is required")
    holdout = verify_fulltrack_test(manifest, lock)
    test_artists = {alias for track in holdout for artist in track.artist_ids for alias in _performer_ids(artist)}
    train_artists = {alias for row in training for alias in _performer_ids(row["artist.name"])}
    train_catalog_artists = {row["artist.id"] for row in training}
    train_works = {row["fma"] for row in training}
    titles = {_identity(track.song_id.rsplit(":", 1)[-1]) for track in holdout}
    titles.update(_identity(row["track.title"].split("(", 1)[0]) for row in training)

    with tarfile.open(audit / METADATA_NAME, "r:gz") as archive:
        train_rows = json.load(archive.extractfile(METADATA_MEMBER))
        valid_rows = json.load(archive.extractfile(VALIDATION_METADATA_MEMBER))
    by_signal = {row["signal"]: row for row in train_rows}
    training_phrases = {tuple(normalize_text(by_signal[row["signal"]]["original_prompt"]).split()) for row in training}
    phrase_lengths = {len(phrase) for phrase in training_phrases}
    references = {f" {track.text} " for track in holdout if not track.diagnostic_only}
    references.update(f" {' '.join(phrase)} " for phrase in training_phrases)

    catalog = _catalog(audit / "fma-tracks.csv", {int(row["fma"]) for row in valid_rows})
    with (audit / LICENSE_NAME).open(encoding="utf-8-sig", newline="") as stream:
        licenses = {int(row["fma_track_id"]): row["licence"].strip() for row in csv.DictReader(stream)}
    candidates = _screen(
        valid_rows, catalog, licenses, test_artists | train_artists, titles, references, allow_noncommercial=True,
    )
    validation_words = {row["signal"]: normalize_text(row["original_prompt"]).split() for row in valid_rows}
    validation = []
    for row in candidates:
        if row["artist.id"] in train_catalog_artists or row["fma"] in train_works:
            continue
        words = validation_words[row["signal"]]
        if any(
            tuple(words[start:start + length]) in training_phrases
            for length in phrase_lengths for start in range(len(words) - length + 1)
        ):
            continue
        validation.append(row)
    if not validation:
        raise ValueError("no independently licensed official CLIP1 validation clips survived screening")

    output.mkdir(parents=True)
    for name in (LICENSE_NAME, METADATA_NAME):
        shutil.copyfile(audit / name, output / name)
    shutil.copyfile(training_audit / "shortlist.jsonl", output / "shortlist.jsonl")
    write_jsonl(output / "validation-shortlist.jsonl", validation)
    return {
        "fixed_train_clips": len(training),
        "validation_clips": len(validation),
        "validation_artists": len({row["artist.id"] for row in validation}),
        "validation_works": len({row["fma"] for row in validation}),
        "validation_genres": dict(Counter(row["track.genre_top"] or "unspecified" for row in validation)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "training_audit", "manifest", "lock", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit, args.training_audit, args.manifest, args.lock, args.output), indent=2))
