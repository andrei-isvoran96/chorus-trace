"""Keep a fixed music subset while adding licensed speech to a scratch experiment."""

import argparse
import os
from pathlib import Path

from chorustrace.data import read_jsonl, write_jsonl
from chorustrace.fulltrack_data import read_fulltrack_development, read_fulltrack_manifest
from chorustrace.fulltrack_training import _training_groups


def combine(music: Path, speech: Path, lock: Path, output: Path, groups: int, seed: int) -> None:
    development = read_fulltrack_development(music, lock)
    auxiliary = read_fulltrack_manifest(speech)
    if not auxiliary or any(track.split != "train" or track.genre != "speech" for track in auxiliary):
        raise ValueError("the auxiliary manifest must contain only training speech")
    partitions = _training_groups([track for track in development if track.split == "train"], seed)
    if not 1 <= groups <= len(partitions):
        raise ValueError(f"music group count must be between 1 and {len(partitions)}")
    retained = {track.track_id for group in partitions[:groups] for track in group}
    rows = []
    for source in (music, speech):
        for _, row in read_jsonl(source):
            if source == music and row["split"] == "train" and row["track_id"] not in retained:
                continue
            for field in ("audio", "reference"):
                row[field] = os.path.relpath(source.parent / row[field], output.parent)
            rows.append(row)
    write_jsonl(output, rows)
    verified = read_fulltrack_development(output, lock)
    training = [track for track in verified if track.split == "train"]
    print(f"{len(retained)} music clips + {len(auxiliary)} speech clips, "
          f"{len(_training_groups(training, seed))} total training groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("music", "speech", "lock", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--music-groups", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    combine(args.music, args.speech, args.lock, args.output, args.music_groups, args.seed)
