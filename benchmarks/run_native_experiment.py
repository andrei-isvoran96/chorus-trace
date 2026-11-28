"""Train and verify native encoders on development audio."""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from chorustrace.audio import FeatureConfig
from chorustrace.data import read_jsonl, split_field, string_field
from chorustrace.fulltrack_data import (
    _audio_signature,
    _check_splits,
    _load_track,
    _read_lock,
    _sha256,
)
from chorustrace.fulltrack_evaluation import evaluate_full_tracks
from chorustrace.fulltrack_training import _training_groups, train_full_tracks
from chorustrace.model import DEFAULT_DILATIONS
from chorustrace.recognizer import Recognizer, TrainingConfig

# Same layers and parameter count, with a wider local frame receptive field.
_ENCODERS = {"baseline": DEFAULT_DILATIONS, "wide-context": (1, 5, 25)}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256(stream)


def _require_unchanged(hashes: dict[Path, str]) -> None:
    for path, digest in hashes.items():
        if _digest(path) != digest:
            raise ValueError(f"{path}: experiment input changed during the run")


def _load_development(path: Path, splits: set[str], lock: Path | None):
    # Check every row before loading any audio or reference from the manifest.
    rows = [(where, row, split_field(row, where, required=True)) for where, row in read_jsonl(path)]
    if {split for _, _, split in rows} != splits:
        raise ValueError(f"{path}: require exactly these development splits: {', '.join(sorted(splits))}")
    entries = [_load_track(path, where, row, split) for where, row, split in rows]
    for (where, row, _), entry in zip(rows, entries, strict=True):
        for field in ("audio_sha256", "pcm_sha256", "reference_sha256"):
            if field in row and row[field] != entry.signature[field]:
                raise ValueError(f"{where}: {field} differs from development assets")
    signatures = [(entry.track.split, entry.signature) for entry in entries]
    if lock is not None:
        signatures.extend(("test", signature) for signature in _read_lock(lock))
    _check_splits(signatures)
    return rows, entries


def _inventory_digest(entries, stems=None, split=None) -> str:
    assets = []
    for entry in sorted(entries, key=lambda entry: entry.track.track_id):
        track = entry.track
        if split is not None and track.split != split:
            continue
        signature = entry.signature
        assets.append((
            track.track_id, signature["audio_sha256"], signature["reference_sha256"],
            stems[track.track_id] if stems is not None else None,
        ))
    return hashlib.sha256(json.dumps(assets, separators=(",", ":")).encode()).hexdigest()


def _run_training(args, entries, lock_hash, checkpoint):
    tracks = [entry.track for entry in entries]
    config = TrainingConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        features=FeatureConfig(
            hop_size=args.hop_size, window_size=args.window_size, stereo_features=args.stereo_features,
        ),
        dilations=_ENCODERS[args.encoder],
        input_deltas=args.input_deltas,
        ctc_prior_weight=args.ctc_prior_weight,
    )
    group_count = len(_training_groups([track for track in tracks if track.split == "train"], config.seed))
    result = train_full_tracks(
        tracks,
        checkpoint,
        config=config,
        scale_groups=(group_count,),
        clip_seconds=args.clip_seconds,
        validate_every=args.validate_every,
        test_lock_sha256=lock_hash,
        progress=lambda progress: print(progress, file=sys.stderr, flush=True),
    )
    result["validation"].pop("samples")
    curve = result["learning_curve"][0]
    prepared = curve["prepared_audio_seconds_by_condition"]
    labeled = curve["labeled_audio_seconds_by_condition"]
    result.update(
        encoder=args.encoder,
        configuration=asdict(config),
        train_inventory_sha256=_inventory_digest(entries, split="train"),
        validation_inventory_sha256=_inventory_digest(entries, split="validation"),
        source_training_audio_hours=curve["training_audio_seconds"] / 3600,
        prepared_training_audio_hours=sum(prepared.values()) / 3600,
        labeled_training_audio_hours_by_condition={key: value / 3600 for key, value in labeled.items()},
        negative_training_audio_hours=(sum(prepared.values()) - sum(labeled.values())) / 3600,
    )
    return result


def _run_paired(args, rows, entries):
    recognizer = Recognizer.load(args.checkpoint)
    mixes = [entry.track for entry in entries]
    overlap = set(recognizer.training_song_ids) & {track.song_id for track in mixes}
    if overlap:
        raise ValueError("paired development songs overlap the checkpoint's training song IDs")
    stems = []
    stem_digests = {}
    all_mix_pcm = {entry.signature["pcm_sha256"] for entry in entries}
    stem_pcm = set()
    for (location, row, _), mix in zip(rows, mixes, strict=True):
        stem = args.manifest.parent / string_field(row, "vocal_audio", location)
        for field in ("vocal_audio_source", "vocal_recording_permission"):
            string_field(row, field, location)
        stem_declarations = {
            field: row.get(f"vocal_{field}", row[field]) for field in ("source_frames", "source_sample_rate")
        }
        duration, digest, pcm = _audio_signature(stem, stem_declarations, location)
        for field, actual in (("vocal_audio_sha256", digest), ("vocal_pcm_sha256", pcm)):
            if field in row and row[field] != actual:
                raise ValueError(f"{location}: {field} differs from stem audio")
        if abs(duration - mix.duration_seconds) > 0.001:
            raise ValueError(f"{location}: stem and full mix cover different durations")
        if pcm in all_mix_pcm or pcm in stem_pcm:
            raise ValueError(f"{location}: stem audio duplicates a mix or another stem")
        stem_digests[mix.track_id] = digest
        stem_pcm.add(pcm)
        stems.append(replace(mix, audio=stem, condition="vocal"))
    versions = {
        condition: evaluate_full_tracks(recognizer, tracks, segment_seconds=args.segment_seconds, samples=0)
        for condition, tracks in (("vocal", stems), ("original_mix", mixes))
    }
    wer = {condition: scores["aggregate"]["wer"] for condition, scores in versions.items()}
    return {
        "paired_inventory_sha256": _inventory_digest(entries, stems=stem_digests),
        "checkpoint_features": asdict(recognizer.features),
        "checkpoint_dilations": recognizer.model.dilations,
        "checkpoint_blank_logit_offset": recognizer.blank_logit_offset,
        "checkpoint_beam_width": recognizer.beam_width,
        "mix_minus_vocal_wer": wer["original_mix"] - wer["vocal"],
        "conditions": versions,
    }


def _score_development(args, entries, lock_hash):
    recognizer = Recognizer.load(args.checkpoint)
    if lock_hash is not None and recognizer.test_lock_sha256 != lock_hash:
        raise ValueError("checkpoint was not trained with the supplied split lock")
    tracks = [entry.track for entry in entries if entry.track.split == "validation"]
    recognizer.check_held_out(tracks)
    return {
        "validation_inventory_sha256": _inventory_digest(entries, split="validation"),
        "validation": evaluate_full_tracks(recognizer, tracks, compare_unsegmented=True, samples=0),
    }


def _write_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace, argv: list[str]) -> dict:
    paths = [args.manifest, args.checkpoint, args.report]
    if args.lock is not None:
        paths.append(args.lock)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("manifest, checkpoint, report, and split lock must use distinct paths")
    start, cpu_start = perf_counter(), process_time()
    training = args.command == "train"
    paired = args.command == "paired"
    inputs = [args.manifest] + ([args.lock] if args.lock is not None else [])
    if not training:
        inputs.append(args.checkpoint)
    hashes = {path: _digest(path) for path in inputs}
    splits = {"validation"} if paired else {"train", "validation"}
    rows, entries = _load_development(args.manifest, splits, args.lock)
    asset_fields = ("audio", "reference", "vocal_audio") if paired else ("audio", "reference")
    assets = {
        (args.manifest.parent / string_field(row, field, location)).resolve()
        for location, row, _ in rows
        for field in asset_fields
    }
    if args.report.resolve() in assets or args.checkpoint.resolve() in assets:
        raise ValueError("checkpoint and report must use paths distinct from development assets")
    lock_hash = hashes.get(args.lock)
    if training:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.checkpoint.parent) as workspace:
            candidate = Path(workspace) / args.checkpoint.name
            result = _run_training(args, entries, lock_hash, candidate)
            _require_unchanged(hashes)
            checkpoint_hash = _digest(candidate)
            os.replace(candidate, args.checkpoint)
    else:
        result = _run_paired(args, rows, entries) if paired else _score_development(args, entries, lock_hash)
        _require_unchanged(hashes)
        checkpoint_hash = hashes[args.checkpoint]
    result.update(
        command=[sys.executable, "-m", "benchmarks.run_native_experiment", *argv],
        working_directory=str(Path.cwd()),
        mode=args.command,
        seed=args.seed if training else None,
        manifest=str(args.manifest.resolve()),
        manifest_sha256=hashes[args.manifest],
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash,
        split_lock=str(args.lock.resolve()) if args.lock is not None else None,
        split_lock_sha256=lock_hash,
        wall_seconds=perf_counter() - start,
        process_cpu_seconds=process_time() - cpu_start,
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        thread_environment={
            name: os.environ[name]
            for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
            if name in os.environ
        },
    )
    _write_report(args.report, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="fit from random initialization")
    paired = subparsers.add_parser("paired", help="score the same checkpoint on matched stem and mix")
    score = subparsers.add_parser("score", help="verify a selected checkpoint's development scores")
    for command in (training, paired, score):
        command.add_argument("manifest", type=Path)
        command.add_argument("checkpoint", type=Path)
        command.add_argument("report", type=Path)
        command.add_argument("--lock", type=Path, help="optional existing split-isolation lock")
    training.add_argument("--encoder", choices=_ENCODERS, default="baseline")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--hidden-size", type=int, default=48)
    training.add_argument("--learning-rate", type=float, default=0.003)
    training.add_argument("--hop-size", type=int, default=320)
    training.add_argument("--window-size", type=int, default=400)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--clip-seconds", type=float, default=12.0)
    training.add_argument("--validate-every", type=int, default=1)
    training.add_argument("--input-deltas", action="store_true")
    training.add_argument("--stereo-features", action="store_true")
    training.add_argument("--ctc-prior-weight", type=float, default=0.0)
    paired.add_argument("--segment-seconds", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run(args, argv)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
