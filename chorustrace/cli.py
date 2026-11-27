"""Command-line training, transcription, evaluation, and fixture preparation."""

import argparse
import hashlib
import json
import sys
import wave
import zipfile
from pathlib import Path

from .audio import FeatureConfig
from .chorus import evaluate_choruses
from .data import SPLITS, read_jsonl, read_manifest, select_split, write_jsonl
from .evaluation import evaluate
from .fulltrack_data import freeze_fulltrack_test, read_fulltrack_development, verify_fulltrack_test
from .fulltrack_evaluation import evaluate_full_tracks
from .fulltrack_training import train_full_tracks
from .model import DEFAULT_DILATIONS
from .phonemes import read_phoneme_supervision
from .musdb import prepare_musdb18
from .prepare import DEFAULT_LEVELS, prepare_benchmark
from .recognizer import Recognizer, TrainingConfig, train
from .synthetic import make_synthetic_sources


def _prepare_synthetic(output: Path, seed: int, levels: tuple[float, ...]) -> Path:
    sources = make_synthetic_sources(output / "sources", seed)
    manifest = prepare_benchmark(sources, output, levels)
    smoke = next(
        row for _, row in read_jsonl(manifest)
        if row["song_id"] == "train-01" and row["clip_id"] == "line-1" and row["condition"] == "vocal"
    )
    write_jsonl(output / "smoke.jsonl", [smoke])
    write_jsonl(output / "repeats.jsonl", [
        {"song_id": "test-01", "clip_a": "repeat-1", "clip_b": "repeat-2"},
        {"song_id": "test-01", "clip_a": "repeat-1", "clip_b": "changed", "changed_lyric": "today"},
    ])
    return manifest


def _split_argument(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--split", choices=(*SPLITS, "all"), default=default,
                        help=f"manifest split to use (default: {default}; use every row if unlabeled)")


def _levels_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--levels", nargs="+", type=float, default=DEFAULT_LEVELS, metavar="DB",
                        help="accompaniment dB relative to vocal RMS (default: -12 -6 0 6)")


def _training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--dilations", type=int, nargs="+", default=DEFAULT_DILATIONS,
                        metavar="FRAMES", help="frame dilation of each acoustic layer")
    parser.add_argument("--input-deltas", action="store_true", help="learn from log-mel slopes and levels")
    parser.add_argument("--stereo-features", action="store_true", help="learn from mid/side spectra in stereo mixes")
    parser.add_argument("--ctc-prior-weight", type=float, default=0.0,
                        help="train with epoch-estimated CTC label priors")
    parser.add_argument("--hop-size", type=int, default=160, help="feature hop in 16 kHz samples")
    parser.add_argument("--window-size", type=int, default=400, help="feature window in 16 kHz samples")
    parser.add_argument("--seed", type=int, default=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chorustrace", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("train", help="fit a fresh model and save a checkpoint")
    fit.add_argument("manifest", type=Path)
    fit.add_argument("checkpoint", type=Path)
    _split_argument(fit, "train")
    fit.add_argument("--condition", action="append", help="train on this condition; repeat to select more")
    _training_arguments(fit)

    decode = commands.add_parser("transcribe", help="print one transcript for a complete PCM WAV")
    decode.add_argument("checkpoint", type=Path)
    decode.add_argument("wav", type=Path)

    score = commands.add_parser("evaluate", help="report WER, CER, and instrumental false alarms as JSON")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("manifest", type=Path)
    _split_argument(score, "test")
    score.add_argument("--held-out", action="store_true", help="check song IDs against checkpoint training IDs")
    score.add_argument("--samples", type=int, default=3, help="sample predictions per condition")

    freeze = commands.add_parser("freeze-test", help="freeze the independently checked test tracks")
    verify = commands.add_parser("verify-test", help="verify frozen test WAVs and references")
    for command in (freeze, verify):
        command.add_argument("manifest", type=Path)
        command.add_argument("lock", type=Path)

    full_fit = commands.add_parser("train-full", help="select a fresh checkpoint on full-track validation")
    full_fit.add_argument("manifest", type=Path)
    full_fit.add_argument("lock", type=Path)
    full_fit.add_argument("checkpoint", type=Path)
    _training_arguments(full_fit)
    full_fit.add_argument("--scale-groups", nargs="+", type=int,
                          help="increasing counts of training identity groups")
    full_fit.add_argument("--clip-seconds", type=float, default=12.0)
    full_fit.add_argument("--validate-every", type=int, default=1)
    full_fit.add_argument("--phoneme-supervision", type=Path)
    full_fit.add_argument("--phoneme-loss-weight", type=float, default=0.0)

    full_score = commands.add_parser("evaluate-full", help="score a saved checkpoint on the frozen test")
    full_score.add_argument("checkpoint", type=Path)
    full_score.add_argument("manifest", type=Path)
    full_score.add_argument("lock", type=Path)
    full_score.add_argument("--samples", type=int, default=3, help="number of sample predictions")

    chorus = commands.add_parser("chorus", help="score independent vs pooled marked chorus takes as JSON")
    chorus.add_argument("checkpoint", type=Path)
    chorus.add_argument("manifest", type=Path)
    chorus.add_argument("pairs", type=Path)
    _split_argument(chorus, "test")
    chorus.add_argument("--held-out", action="store_true", help="check song IDs against checkpoint training IDs")
    chorus.add_argument("--condition", default="original_mix")

    prepare = commands.add_parser("prepare", help="render benchmarks from aligned stems in a source manifest")
    prepare.add_argument("sources", type=Path)
    prepare.add_argument("output", type=Path)
    _levels_argument(prepare)

    synthetic = commands.add_parser("prepare-synthetic", help="generate offline sung fixtures and pair marks")
    synthetic.add_argument("output", type=Path)
    synthetic.add_argument("--seed", type=int, default=0)
    _levels_argument(synthetic)

    musdb = commands.add_parser("prepare-musdb18", help="render selected clips from local MUSDB18-HQ WAVs")
    musdb.add_argument("audio", type=Path)
    musdb.add_argument("lyrics", type=Path)
    musdb.add_argument("selection", type=Path)
    musdb.add_argument("output", type=Path)
    musdb.add_argument("--mix-only", action="store_true", help="render only the original mixture")
    _levels_argument(musdb)
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        print(prepare_benchmark(args.sources, args.output, args.levels))
    elif args.command == "prepare-synthetic":
        print(_prepare_synthetic(args.output, args.seed, args.levels))
    elif args.command == "prepare-musdb18":
        print(prepare_musdb18(args.audio, args.lyrics, args.selection, args.output,
                              levels=args.levels, mix_only=args.mix_only))
    elif args.command == "transcribe":
        print(Recognizer.load(args.checkpoint).transcribe(args.wav))
    elif args.command == "freeze-test":
        freeze_fulltrack_test(args.manifest, args.lock)
        print(args.lock)
    elif args.command == "verify-test":
        tracks = verify_fulltrack_test(args.manifest, args.lock)
        diagnostics = sum(track.diagnostic_only for track in tracks)
        summary = {"tracks": len(tracks) - diagnostics, "test_lock_sha256": _sha256(args.lock)}
        if diagnostics:
            summary["instrumental_diagnostic_tracks"] = diagnostics
        print(json.dumps(summary, indent=2))
    elif args.command == "train-full":
        lock_digest = _sha256(args.lock)
        tracks = read_fulltrack_development(args.manifest, args.lock)
        if _sha256(args.lock) != lock_digest:
            raise ValueError("the test lock changed during development data verification")
        phonemes = read_phoneme_supervision(args.phoneme_supervision, tracks) if args.phoneme_supervision else None
        report = train_full_tracks(
            tracks, args.checkpoint, config=_training_config(args),
            scale_groups=args.scale_groups, clip_seconds=args.clip_seconds,
            validate_every=args.validate_every, test_lock_sha256=lock_digest,
            phonemes=phonemes,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        if _sha256(args.lock) != lock_digest:
            raise ValueError("the test lock changed during training")
        report["test_lock_sha256"] = lock_digest
        report["checkpoint_sha256"] = _sha256(args.checkpoint)
        print(json.dumps(report, indent=2))
    elif args.command == "evaluate-full":
        lock_digest = _sha256(args.lock)
        checkpoint_digest = _sha256(args.checkpoint)
        recognizer = Recognizer.load(args.checkpoint)
        if recognizer.test_lock_sha256 != lock_digest:
            raise ValueError("checkpoint was not trained with this frozen test lock")
        tracks = verify_fulltrack_test(args.manifest, args.lock)
        if _sha256(args.lock) != lock_digest or _sha256(args.checkpoint) != checkpoint_digest:
            raise ValueError("the test lock or checkpoint changed during verification")
        recognizer.check_held_out(tracks)
        report = evaluate_full_tracks(recognizer, tracks, samples=args.samples, compare_unsegmented=True)
        verify_fulltrack_test(args.manifest, args.lock)
        if _sha256(args.lock) != lock_digest or _sha256(args.checkpoint) != checkpoint_digest:
            raise ValueError("the test lock or checkpoint changed during evaluation")
        report["test_lock_sha256"] = lock_digest
        report["checkpoint_sha256"] = checkpoint_digest
        report["meets_goal"] = report["track_count"] >= 20 and report["aggregate"]["wer"] <= 0.02
        print(json.dumps(report, indent=2))
    else:
        examples = select_split(read_manifest(args.manifest), args.split)
        if args.command == "train":
            _train(args, examples)
        else:
            recognizer = Recognizer.load(args.checkpoint)
            if args.held_out:
                recognizer.check_held_out(examples)
            if args.command == "evaluate":
                report = evaluate(recognizer, examples, args.samples)
            else:
                report = evaluate_choruses(recognizer, examples, args.pairs, args.condition)
            print(json.dumps(report, indent=2))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(args.epochs, args.learning_rate, args.hidden_size, args.seed,
        FeatureConfig(window_size=args.window_size, hop_size=args.hop_size,
                      stereo_features=args.stereo_features),
                          dilations=tuple(args.dilations),
                          input_deltas=args.input_deltas, ctc_prior_weight=args.ctc_prior_weight,
                          phoneme_loss_weight=getattr(args, "phoneme_loss_weight", 0.0))


def _train(args: argparse.Namespace, examples: list) -> None:
    if args.condition:
        missing = set(args.condition) - {example.condition for example in examples}
        if missing:
            raise ValueError(f"training split lacks conditions: {', '.join(sorted(missing))}")
        examples = [example for example in examples if example.condition in args.condition]
    config = _training_config(args)
    songs = len({example.song_id for example in examples})
    print(f"training on {len(examples)} clips from {songs} songs", file=sys.stderr, flush=True)

    def progress(epoch: int, loss: float) -> None:
        if epoch == 1 or epoch == config.epochs or epoch % 5 == 0:
            print(f"epoch {epoch}/{config.epochs}  CTC loss/frame {loss:.5f}", file=sys.stderr, flush=True)

    train(examples, config, progress=progress).save(args.checkpoint)
    print(args.checkpoint)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (OSError, ValueError, TypeError, KeyError, EOFError, wave.Error, zipfile.BadZipFile) as error:
        parser.exit(1, f"chorustrace: {error}\n")
    return 0
