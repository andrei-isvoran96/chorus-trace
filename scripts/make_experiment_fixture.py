"""Prepare procedural singing fixtures for the frozen experiment workflow."""

import argparse
import wave
from pathlib import Path

from chorustrace.data import read_jsonl, write_jsonl
from chorustrace.synthetic import make_synthetic_sources
from scripts.prepare_lyrics import freeze


def make_fixture(output: Path, seed: int = 0) -> Path:
    output.mkdir(parents=True)
    raw = make_synthetic_sources(output / "sources", seed)
    sources = raw.with_name("verified.jsonl")
    rights = output / "rights.jsonl"

    rows = []
    for _, row in read_jsonl(raw):
        song = row["song_id"]
        with wave.open(str(raw.parent / row["vocal"]), "rb") as vocal:
            duration = vocal.getnframes() / vocal.getframerate()
        rows.append(dict(
            row,
            split="validation" if song in ("train-05", "train-06") else row["split"],
            artist=f"procedural-{song}",
            genre="synthetic",
            recording_id=f"{song}/{row['clip_id']}",
            language="en",
            start=0,
            end=duration,
            rights_id="procedural-fixture",
            annotation_verified=True,
            alignment_verified=True,
        ))
    write_jsonl(sources, rows)
    write_jsonl(rights, [{
        "rights_id": "procedural-fixture",
        "source": "chorustrace/synthetic.py (project-authored procedural fixtures)",
        "recording_permission": "Project-generated synthetic audio with no third-party recordings",
        "lyrics_permission": "Project-authored fixture text",
        "training_allowed": True,
        "checkpoint_release_allowed": True,
    }])
    return freeze(sources, rights, output / "prepared", seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        print(make_fixture(args.output, args.seed))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
