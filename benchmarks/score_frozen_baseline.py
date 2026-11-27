"""Measure the pre-goal generated checkpoint on the frozen release, before real training."""

import argparse
import hashlib
import json
from pathlib import Path

from chorustrace.fulltrack_data import verify_fulltrack_test
from chorustrace.fulltrack_evaluation import evaluate_full_tracks
from chorustrace.recognizer import Recognizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("lock", type=Path)
    args = parser.parse_args()
    checkpoint = Path(__file__).resolve().parent / "checkpoints" / "synthetic_baseline.npz"
    lock_digest = sha256(args.lock)
    tracks = verify_fulltrack_test(args.manifest, args.lock)
    recognizer = Recognizer.load(checkpoint)
    recognizer.check_held_out(tracks)
    report = evaluate_full_tracks(recognizer, tracks, samples=0, compare_unsegmented=True)
    verify_fulltrack_test(args.manifest, args.lock)
    if sha256(args.lock) != lock_digest:
        raise ValueError("the test lock changed while evaluating the legacy baseline")
    report.update(scope="pre-goal generated-audio checkpoint on the frozen release",
                  test_lock_sha256=lock_digest, checkpoint_sha256=sha256(checkpoint))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
