"""Join staged CLIP1 examples while retaining shared work and performer IDs."""

import argparse
import os
from pathlib import Path

from benchmarks.screen_clip1_development import _performer_ids
from chorustrace.data import read_jsonl, write_jsonl


def combine(audit: Path, train: Path, validation: Path, destination: Path) -> None:
    combined = []
    for source, split, catalogue in ((train, "train", "shortlist.jsonl"),
                                     (validation, "validation", "validation-shortlist.jsonl")):
        wanted = {row["signal"]: row for _, row in read_jsonl(audit / catalogue)}
        found = set()
        for _, row in read_jsonl(source):
            id_prefix, _, signal = row["track_id"].rpartition(":")
            entry = wanted.get(signal)
            if entry is None:
                continue
            if (id_prefix != "clip1" or row["split"] != split or signal in found
                    or row["song_id"] != f"fma:{entry['fma']}"):
                raise ValueError(f"{source}: unexpected staged identity, split or duplicate signal {signal}")
            found.add(signal)
            for field in ("audio", "reference"):
                row[field] = os.path.relpath(source.parent / row[field], destination.parent)
            row["artist_ids"] = sorted(set(row["artist_ids"]) | _performer_ids(entry["artist.name"]))
            if row["song_id"] in {"fma:12240", "fma:12264"}:
                row["song_id"] = row["lyric_id"] = "foot-village:trumpet"
            combined.append(row)
        if found != set(wanted):
            raise ValueError(f"{source}: missing {len(set(wanted) - found)} screened signals")
    write_jsonl(destination, combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "train", "validation", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    combine(args.audit, args.train, args.validation, args.destination)
