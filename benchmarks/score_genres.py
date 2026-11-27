"""Score one checkpoint on a common holdout and on each marked genre."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from chorustrace import Recognizer, evaluate, read_manifest
from chorustrace.data import read_jsonl, select_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("marks", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    genres = {}
    for location, row in read_jsonl(args.marks):
        song, genre = row["song_id"], row["genre"]
        if genres.setdefault(song, genre) != genre:
            raise ValueError(f"{location}: conflicting genres for {song!r}")

    examples = select_split(read_manifest(args.manifest), "test")
    recognizer = Recognizer.load(args.checkpoint)
    recognizer.check_held_out(examples)
    groups = defaultdict(list)
    for example in examples:
        groups[genres[example.song_id]].append(example)

    print(json.dumps({
        "all": evaluate(recognizer, examples, samples=args.samples),
        "genres": {
            genre: evaluate(recognizer, group, samples=args.samples)
            for genre, group in sorted(groups.items())
        },
    }, indent=2))


if __name__ == "__main__":
    main()
