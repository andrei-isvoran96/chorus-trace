"""Choose a trained checkpoint and decoder blank penalty on development data only."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from chorustrace.fulltrack_data import read_fulltrack_development
from chorustrace.fulltrack_evaluation import evaluate_full_tracks
from chorustrace.recognizer import Recognizer


OFFSETS = (0, 0.5, 1, 1.5, 2, 3, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()

    lock_digest = hashlib.sha256(args.lock.read_bytes()).hexdigest()
    tracks = read_fulltrack_development(args.manifest, args.lock)
    validation = [track for track in tracks if track.split == "validation"]
    best, best_key, best_model, iterations = None, None, None, []
    for path in args.candidates:
        model = Recognizer.load(path)
        if model.test_lock_sha256 != lock_digest:
            raise ValueError(f"{path}: checkpoint was fitted against a different frozen test lock")
        model.check_held_out(validation)
        for offset in OFFSETS:
            candidate = replace(model, blank_logit_offset=offset)
            scores = evaluate_full_tracks(candidate, validation, samples=0)["aggregate"]
            row = {"candidate": path.name, "blank_logit_offset": offset, **scores}
            iterations.append(row)
            key = (scores["wer"], scores["cer"], offset)
            if best_key is None or key < best_key:
                best_key, best, best_model = key, row, candidate

    if hashlib.sha256(args.lock.read_bytes()).hexdigest() != lock_digest:
        raise ValueError("the test lock changed during validation")
    best_model.save(args.output)
    print(json.dumps({"scope": "artist-disjoint development excerpts only",
                      "selection": best, "sweep": iterations,
                      "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
