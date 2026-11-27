"""Select an acoustic checkpoint and CTC search using only fixed development data."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from benchmarks.prepare_rwc_instrumental_development import read_bank
from benchmarks.screen_clip1_development import _performer_ids
from chorustrace.data import read_jsonl
from chorustrace.fulltrack_data import _ID_FIELDS, _identity, _read_lock, read_fulltrack_development
from chorustrace.fulltrack_evaluation import evaluate_full_tracks
from chorustrace.recognizer import Recognizer


_COUNTS = ("reference_words", "reference_characters", "substitutions", "deletions", "insertions",
           "word_edits", "character_edits")
_DECODERS = ((1, 0.5), (1, 1.0), (4, 0.0), (8, 0.0), (8, 0.5), (8, 1.0))
_GROUP_FIELDS = (*_ID_FIELDS, "audio_sha256", "pcm_sha256")


def _scores(rows: list[dict]) -> dict:
    sums = {name: sum(row[name] for row in rows) for name in _COUNTS}
    return sums | {"tracks": len(rows), "wer": sums["word_edits"] / sums["reference_words"],
                   "cer": sums["character_edits"] / sums["reference_characters"],
                   "emitted_words": sums["reference_words"] - sums["deletions"] + sums["insertions"]}


def _groups(row: dict) -> set[tuple[str, str]]:
    groups = {(field, _identity(row[field])) for field in _GROUP_FIELDS if row.get(field)}
    groups.update(("artist_ids", _identity(credit)) for credit in row["artist_ids"])
    groups.update(("artist_ids", person) for credit in row["artist_ids"] if ":" not in credit
                  for person in _performer_ids(credit))
    return groups


def _check_bank_isolation(bank: Path, development: Path, lock: Path) -> None:
    protected = {key for item in _read_lock(lock) for key in _groups(item)}
    protected.update(key for _, row in read_jsonl(development)
                     if row.get("split") in ("train", "validation") for key in _groups(row))
    seen = set()
    for location, row in read_jsonl(bank):
        groups = _groups(row)
        if groups & protected:
            raise ValueError(f"{location}: instrumental bank overlaps the development or frozen identities")
        audio = {key for key in groups if key[0] in ("track_id", "audio_sha256", "pcm_sha256")}
        if audio & seen:
            raise ValueError(f"{location}: duplicate instrumental diagnostic audio")
        seen.update(audio)


def _frontier(rows: list[dict]) -> list[dict]:
    points = [(row["wer"], row["cer"], row["instrumental"]["words_per_minute"]) for row in rows]
    return [row for row, point in zip(rows, points, strict=True)
            if not any(other != point and all(left <= right for left, right in zip(other, point, strict=True))
                       for other in points)]


def compare(args: argparse.Namespace) -> dict:
    cap = getattr(args, "max_instrumental_words_per_minute", None)
    joint = getattr(args, "joint_instrumental", False) or cap is not None
    if cap is not None and (not math.isfinite(cap) or cap < 0):
        raise ValueError("instrumental false-word cap must be finite and nonnegative")
    if joint and args.instrumental is None:
        raise ValueError("joint validation requires a development instrumental bank")
    lock_hash = hashlib.sha256(args.lock.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    bank_hash = hashlib.sha256(args.instrumental.read_bytes()).hexdigest() if args.instrumental else None
    if args.instrumental:
        _check_bank_isolation(args.instrumental, args.manifest, args.lock)
    tracks = read_fulltrack_development(args.manifest, args.lock)
    validation = [track for track in tracks if track.split == "validation"]
    diagnostics = read_bank(args.instrumental) if args.instrumental else []

    def evaluate(path, model, width, offset):
        configured = replace(model, beam_width=width, blank_logit_offset=offset)
        report = evaluate_full_tracks(configured, validation + diagnostics if joint else validation, samples=0)
        result = {"candidate": path.name, "beam_width": width, "blank_logit_offset": offset,
                  **_scores(report["tracks"])}
        if joint:
            result["instrumental"] = report["instrumental_diagnostics"]["instrumental"]
        return result, configured

    def ranking(item):
        row, _ = item
        return row["wer"], row["cer"], row["beam_width"], row["blank_logit_offset"]

    candidates, greedy = [], []
    for path in args.candidates:
        model = Recognizer.load(path)
        if model.test_lock_sha256 != lock_hash:
            raise ValueError(f"{path}: checkpoint was fitted against a different frozen test lock")
        model.check_held_out(validation + diagnostics)
        candidates.append((path, model))
        greedy.append(evaluate(path, model, 1, 0.0))
    order = sorted(range(len(greedy)), key=lambda index: ranking(greedy[index]))
    all_scores = list(greedy)
    for index in order[:2]:
        for width, offset in _DECODERS:
            all_scores.append(evaluate(*candidates[index], width, offset))
    eligible = [item for item in all_scores if cap is None or item[0]["instrumental"]["words_per_minute"] <= cap]
    winner, model = min(eligible, default=(None, None), key=ranking)
    selected = evaluate_full_tracks(model, validation + diagnostics, samples=3) if model is not None else None
    genre = defaultdict(list)
    for track in selected["tracks"] if selected else []:
        genre[track["genre"]].append(track)
    baseline = (evaluate_full_tracks(candidates[0][1], validation + diagnostics, samples=0)
                if diagnostics else {})
    if hashlib.sha256(args.lock.read_bytes()).hexdigest() != lock_hash:
        raise ValueError("the test lock changed during validation")
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("the development manifest changed during validation")
    if args.instrumental and hashlib.sha256(args.instrumental.read_bytes()).hexdigest() != bank_hash:
        raise ValueError("the development instrumental manifest changed during validation")
    if model is not None:
        model.save(args.output)
    sweep = [row for row, _ in all_scores]
    report = {
        "scope": "development only",
        "test_lock_sha256": lock_hash,
        "development_manifest_sha256": manifest_hash,
        "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest() if model is not None else None,
        "selection": winner,
        "sweep": sweep,
        "selected_by_genre": {name: _scores(rows) for name, rows in sorted(genre.items())},
        "selected": selected,
        "incumbent_instrumental": baseline.get("instrumental_diagnostics"),
    }
    if joint:
        report["joint_validation"] = {
            "instrumental_manifest_sha256": bank_hash,
            "max_instrumental_words_per_minute": cap,
            "eligible_configurations": len(eligible),
            "instrumental_tracks": len(diagnostics),
            "instrumental_genres": sorted({track.genre for track in diagnostics}),
            "selection_order": ["wer", "cer", "beam_width", "blank_logit_offset"],
            "frontier_metrics": ["wer", "cer", "instrumental.words_per_minute"],
            "frontier": _frontier(sweep),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--instrumental", type=Path, help="separate development instrumental manifest")
    parser.add_argument("--joint-instrumental", action="store_true", help="measure false alarms for every tested setup")
    parser.add_argument("--max-instrumental-words-per-minute", type=float,
                        help="optional development false-word cap for checkpoint selection")
    args = parser.parse_args()
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()
