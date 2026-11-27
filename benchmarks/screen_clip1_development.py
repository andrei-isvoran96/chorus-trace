"""Screen CLIP1 development excerpts against the independently frozen holdout."""

import argparse
import csv
import json
import re
import shutil
import tarfile
from pathlib import Path

from benchmarks.prepare_clip1_training import LICENSE_NAME, METADATA_NAME, is_allowed_license
from chorustrace.data import normalize_text, read_jsonl, write_jsonl
from chorustrace.fulltrack_data import verify_fulltrack_test


def _identity(value: str) -> str:
    return "".join(char for char in normalize_text(value) if char.isalnum())


def _performer_ids(credit: str) -> set[str]:
    parts = re.split(r"\s+(?:[&+/;]|feat\.?|featuring|ft\.?|with)\s+|,\s+", credit, flags=re.IGNORECASE)
    return {_identity(part) for part in (*parts, credit)} - {""}


def _catalog(path: Path, wanted: set[int]) -> dict:
    csv.field_size_limit(10_000_000)
    fields = (("artist", "id"), ("artist", "name"), ("track", "title"),
              ("track", "genre_top"), ("track", "license"))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        columns = {field: index for index, field in enumerate(zip(next(rows), next(rows)))}
        next(rows)
        return {int(row[0]): {".".join(field): row[columns[field]] for field in fields}
                for row in rows if int(row[0]) in wanted}


def _screen(rows, catalog, licenses, artists, titles, references, signals=None, *,
            allow_noncommercial=False):
    selected = []
    seen = set()
    reference_phrases = set()
    for reference in references:
        words = reference.split()
        reference_phrases.update(tuple(words[index:index + 6]) for index in range(len(words) - 5))
    for row in rows:
        signal, fma = row["signal"], int(row["fma"])
        raw = row["original_prompt"]
        if (signals is not None and signal not in signals) or row["hearing_loss"] != "No Loss" or "?" in raw:
            continue
        track, license_name = catalog.get(fma), licenses.get(fma)
        if (track is None or not is_allowed_license(license_name, allow_noncommercial=allow_noncommercial)
                or track["track.license"] != license_name):
            continue
        if (_performer_ids(track["artist.name"]) & artists
                or _identity(track["track.title"].split("(", 1)[0]) in titles):
            continue
        prompt = normalize_text(raw)
        words = prompt.split()
        if (not prompt or any(f" {prompt} " in reference for reference in references)
                or any(tuple(words[index:index + 6]) in reference_phrases for index in range(len(words) - 5))):
            continue
        if (fma, prompt) in seen or track.get("track.genre_top") == "Spoken":
            continue
        seen.add((fma, prompt))
        selected.append({"signal": signal, "fma": fma, "clip1_license": license_name, **track})
    return selected


def prepare(audit: Path, manifest: Path, lock: Path, output: Path, *,
            baseline: Path | None = None, allow_noncommercial: bool = False) -> tuple[int, int]:
    holdout = verify_fulltrack_test(manifest, lock)
    performers = {artist for track in holdout for artist in track.artist_ids}
    titles = {track.song_id.split(":")[-1] for track in holdout}
    references = [f" {track.text} " for track in holdout if not track.diagnostic_only]
    with tarfile.open(audit / METADATA_NAME) as archive:
        train, valid = (json.load(archive.extractfile(f"cadenza_data/metadata/{split}_metadata_with_phoneme.json"))
                        for split in ("train", "valid"))
    catalog = _catalog(audit / "fma-tracks.csv", {int(row["fma"]) for row in train + valid})
    with (audit / LICENSE_NAME).open(encoding="utf-8-sig") as stream:
        licenses = {int(row["fma_track_id"]): row["licence"].strip() for row in csv.DictReader(stream)}
    if baseline is None:
        existing = {row["signal"] for _, row in read_jsonl(audit / "shortlist.jsonl")}
        selected_train = _screen(train, catalog, licenses, performers, titles, references, existing)
        training_artists = set().union(*(_performer_ids(row["artist.name"]) for row in selected_train))
        selected_valid = _screen(valid, catalog, licenses,
                                 performers | training_artists, titles, references)
    else:
        selected_valid = [row for _, row in read_jsonl(baseline / "validation-shortlist.jsonl")]
        validation_artists = set().union(*(_performer_ids(row["artist.name"]) for row in selected_valid))
        existing = None if allow_noncommercial else {row["signal"] for _, row in read_jsonl(audit / "shortlist.jsonl")}
        selected_train = _screen(train, catalog, licenses, performers | validation_artists, titles,
                                 references, existing, allow_noncommercial=allow_noncommercial)
    if not selected_train or not selected_valid:
        raise ValueError("no separately licensed training or validation excerpts survived screening")
    output.mkdir(parents=True)
    for name in (LICENSE_NAME, METADATA_NAME):
        shutil.copyfile(audit / name, output / name)
    write_jsonl(output / "shortlist.jsonl", selected_train)
    write_jsonl(output / "validation-shortlist.jsonl", selected_valid)
    return len(selected_train), len(selected_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path,
                        help="keep the baseline validation fixed while screening training")
    parser.add_argument("--allow-noncommercial", action="store_true",
                        help="admit additional training excerpts licensed for noncommercial research")
    arguments = parser.parse_args()
    if arguments.allow_noncommercial and arguments.baseline is None:
        parser.error("--allow-noncommercial requires --baseline to keep validation fixed")
    training, validation = prepare(arguments.audit, arguments.manifest, arguments.lock, arguments.output,
                                   baseline=arguments.baseline, allow_noncommercial=arguments.allow_noncommercial)
    print(f"{training} training and {validation} validation excerpts after the holdout audit")
