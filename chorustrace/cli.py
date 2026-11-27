"""Command-line training, transcription, evaluation, and fixture preparation."""

import argparse
import json
import sys
import wave
import zipfile
from pathlib import Path

from .abstention import load_threshold
from .audio import FeatureConfig
from .chorus import evaluate_choruses
from .data import (
    SPLITS,
    file_sha256,
    read_jsonl,
    read_manifest,
    select_split,
    write_jsonl,
)
from .evaluation import evaluate
from .model import MODEL_ARCHITECTURES
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chorustrace", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("train", help="fit a fresh model and save a checkpoint")
    fit.add_argument("manifest", type=Path)
    fit.add_argument("checkpoint", type=Path)
    _split_argument(fit, "train")
    fit.add_argument("--condition", action="append", help="train on this condition; repeat to select more")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--learning-rate", type=float, default=0.003)
    fit.add_argument("--hidden-size", type=int, default=48)
    fit.add_argument("--hop-size", type=int, default=160, help="feature hop in 16 kHz samples")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--architecture", choices=MODEL_ARCHITECTURES, default="baseline")

    decode = commands.add_parser("transcribe", help="print a greedy transcript of one WAV")
    decode.add_argument("checkpoint", type=Path)
    decode.add_argument("wav", type=Path)
    decode.add_argument("--abstention", type=Path, help="apply a checkpoint-specific calibration JSON")

    score = commands.add_parser("evaluate", help="report WER, CER, and instrumental false alarms as JSON")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("manifest", type=Path)
    _split_argument(score, "test")
    score.add_argument("--held-out", action="store_true", help="check song IDs against checkpoint training IDs")
    score.add_argument("--samples", type=int, default=3, help="sample predictions per condition")
    score.add_argument("--abstention", type=Path, help="apply a checkpoint-specific calibration JSON")

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
        print(Recognizer.load(args.checkpoint).transcribe(args.wav, threshold=_threshold(args)))
    else:
        examples = select_split(read_manifest(args.manifest), args.split)
        if args.command == "train":
            _train(args, examples)
        else:
            recognizer = Recognizer.load(args.checkpoint)
            if args.held_out:
                recognizer.check_held_out(examples)
            if args.command == "evaluate":
                report = evaluate(recognizer, examples, args.samples, threshold=_threshold(args))
            else:
                report = evaluate_choruses(recognizer, examples, args.pairs, args.condition)
            print(json.dumps(report, indent=2))


def _threshold(args: argparse.Namespace) -> float | None:
    if args.abstention is None:
        return None
    return load_threshold(args.abstention, args.checkpoint)


def _train(args: argparse.Namespace, examples: list) -> None:
    if args.condition:
        missing = set(args.condition) - {example.condition for example in examples}
        if missing:
            raise ValueError(f"training split lacks conditions: {', '.join(sorted(missing))}")
        examples = [example for example in examples if example.condition in args.condition]
    config = TrainingConfig(args.epochs, args.learning_rate, args.hidden_size, args.seed,
                            FeatureConfig(hop_size=args.hop_size), architecture=args.architecture)
    songs = len({example.song_id for example in examples})
    print(f"training on {len(examples)} clips from {songs} songs", file=sys.stderr, flush=True)

    def progress(epoch: int, loss: float) -> None:
        if epoch == 1 or epoch == config.epochs or epoch % 5 == 0:
            print(f"epoch {epoch}/{config.epochs}  CTC loss/frame {loss:.5f}", file=sys.stderr, flush=True)

    manifest_sha256 = file_sha256(args.manifest)
    train(examples, config, progress=progress, training_manifest_sha256=manifest_sha256).save(args.checkpoint)
    print(args.checkpoint)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (OSError, ValueError, TypeError, KeyError, EOFError, wave.Error, zipfile.BadZipFile) as error:
        parser.exit(1, f"chorustrace: {error}\n")
    return 0
