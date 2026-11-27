"""Fit nested isolated-vocal subsets against one frozen validation split."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from bisect import bisect_left
from dataclasses import asdict
from itertools import accumulate
from pathlib import Path
from time import perf_counter

from chorustrace.audio import FeatureConfig
from chorustrace.data import file_sha256, read_jsonl, read_manifest, write_jsonl
from chorustrace.recognizer import Recognizer, TrainingConfig, train
from scripts.prepare_lyrics import verify_frozen


def _subset_sizes(durations: list[float], requested_hours: list[float]) -> list[tuple[float | None, int]]:
    cumulative = list(accumulate(durations))
    if not cumulative:
        raise ValueError("the training split has no isolated vocals")
    if not requested_hours or any(not math.isfinite(hours) or hours <= 0 for hours in requested_hours):
        raise ValueError("training hour targets must be positive and finite")

    rungs = []
    for hours in sorted(set(requested_hours)):
        size = bisect_left(cumulative, hours * 3600) + 1
        if size > len(cumulative):
            if not rungs or rungs[-1][1] < len(cumulative):
                rungs.append((None, len(cumulative)))
            break
        rungs.append((hours, size))
    return rungs


def _write_subset(path: Path, prepared: Path, rows: list[dict]) -> None:
    write_jsonl(path, (dict(row, audio=os.path.relpath(prepared.parent / row["audio"], path.parent))
                       for row in rows))


def _score(checkpoint: Path, manifest: Path, split: str) -> dict:
    command = [sys.executable, "-m", "chorustrace", "evaluate", str(checkpoint), str(manifest),
               "--split", split, "--samples", "0"]
    if split == "validation":
        command.append("--held-out")
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(result.stdout)["conditions"]["vocal"]


def run_curve(prepared: Path, output: Path, config: TrainingConfig, requested_hours: list[float]) -> Path:
    manifest = verify_frozen(prepared)
    manifest_sha = file_sha256(manifest)
    vocal_rows = [row for _, row in read_jsonl(manifest) if row["condition"] == "vocal"]
    training = sorted((row for row in vocal_rows if row["split"] == "train"),
                      key=lambda row: (row["song_id"], row["clip_id"]))
    validation = [row for row in vocal_rows if row["split"] == "validation"]
    if not validation:
        raise ValueError("the fixed validation split has no isolated vocals")
    if any(not row["text"] for row in training + validation):
        raise ValueError("each isolated vocal needs a lyric reference")
    random.Random(config.seed).shuffle(training)
    durations = [row["duration_seconds"] for row in training]
    sizes = _subset_sizes(durations, requested_hours)

    output.mkdir(parents=True)
    validation_path = output / "validation.jsonl"
    _write_subset(validation_path, manifest, validation)
    report = {
        "manifest_sha256": manifest_sha,
        "validation_manifest": validation_path.name,
        "validation_sha256": file_sha256(validation_path),
        "validation_seconds": math.fsum(row["duration_seconds"] for row in validation),
        "validation_clips": len(validation),
        "available_training_seconds": math.fsum(durations),
        "requested_hours": sorted(set(requested_hours)),
        "config": asdict(config),
        "rungs": [],
    }
    for hours, size in sizes:
        name = f"{hours}h" if hours is not None else "available"
        subset_path = output / f"{name}.jsonl"
        checkpoint_path = output / f"{name}.npz"
        _write_subset(subset_path, manifest, training[:size])
        clips = read_manifest(subset_path)
        losses = []

        def progress(epoch: int, loss: float, *, recorded=losses, rung=name) -> None:
            recorded.append(loss)
            if epoch in (1, config.epochs) or epoch % 5 == 0:
                print(f"{rung} epoch {epoch}/{config.epochs}: {loss:.4f} CTC loss/frame", file=sys.stderr)

        started = perf_counter()
        recognizer = train(clips, config, progress=progress, training_manifest_sha256=manifest_sha)
        training_seconds = perf_counter() - started
        recognizer.save(checkpoint_path)
        restored = Recognizer.load(checkpoint_path)
        restored.check_held_out(read_manifest(validation_path))
        if restored.training_manifest_sha256 != manifest_sha:
            raise ValueError("checkpoint did not preserve the frozen manifest digest")
        scored = perf_counter()
        training_scores = _score(checkpoint_path, subset_path, "train")
        validation_scores = _score(checkpoint_path, validation_path, "validation")
        audio_seconds = math.fsum(durations[:size])
        report["rungs"].append({
            "target_hours": hours,
            "audio_seconds": audio_seconds,
            "audio_hours": audio_seconds / 3600,
            "training_clips": size,
            "optimizer_steps": size * config.epochs,
            "fit_wall_seconds": training_seconds,
            "score_wall_seconds": perf_counter() - scored,
            "loss_per_frame_by_epoch": losses,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "training_manifest": subset_path.name,
            "training_manifest_sha256": file_sha256(subset_path),
            "training_song_ids": list(restored.training_song_ids),
            "training_artist_ids": list(restored.training_artist_ids),
            "training_scores": training_scores,
            "validation_scores": validation_scores,
        })
    path = output / "curve.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, help="frozen prepared dataset directory")
    parser.add_argument("output", type=Path, help="new directory for subset manifests, scores, and checkpoints")
    parser.add_argument("--hours", nargs="+", type=float, default=[1, 5, 10])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--architecture", choices=("baseline", "long-context"), default="baseline")
    args = parser.parse_args()
    try:
        config = TrainingConfig(epochs=args.epochs, learning_rate=args.learning_rate, hidden_size=args.hidden_size,
                                seed=args.seed, features=FeatureConfig(hop_size=args.hop_size),
                                architecture=args.architecture)
        print(run_curve(args.prepared, args.output, config, args.hours))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
