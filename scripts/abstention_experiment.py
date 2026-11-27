"""Fit a greedy CTC confidence gate on paired validation clips."""

import argparse
import json
import os
import sys
import tempfile
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chorustrace.abstention import (
    CALIBRATION_VERSION,
    CONFIDENCE_METHOD,
    abstain,
    decode_with_confidence,
)
from chorustrace.data import Example, file_sha256, read_manifest
from chorustrace.evaluation import ErrorCounts
from chorustrace.recognizer import Recognizer


@dataclass(frozen=True)
class ScoredClip:
    reference: str
    prediction: str
    confidence: float


def select_threshold(instrumental_confidences: Sequence[float], max_false_alarm_rate: float) -> float:
    """Keep the most voiced output subject to a strict validation false-alarm limit."""
    total = len(instrumental_confidences)
    if not total:
        raise ValueError("calibration requires paired validation instrumentals")
    if not 0 < max_false_alarm_rate <= 1:
        raise ValueError("maximum false-alarm rate must be in (0, 1]")
    allowed = bisect_left(range(total + 1), max_false_alarm_rate, key=lambda count: count / total) - 1
    return sorted(instrumental_confidences, reverse=True)[allowed]


def _group_validation(examples: Sequence[Example]) -> dict[tuple[str, str], dict[str, Example]]:
    groups: dict[tuple[str, str], dict[str, Example]] = {}
    for example in examples:
        if not example.clip_id:
            raise ValueError("validation rows require clip_id for condition pairing")
        key = (example.song_id, example.clip_id)
        group = groups.setdefault(key, {})
        if example.condition in group:
            raise ValueError(f"duplicate {example.condition!r} condition for validation clip {key}")
        group[example.condition] = example
        expected_text = example.condition != "instrumental"
        if bool(example.text) != expected_text:
            raise ValueError(f"invalid reference for {example.condition!r} validation clip {key}")
    for key, group in groups.items():
        voiced_references = {row.text for row in group.values() if row.text}
        if len(voiced_references) > 1:
            raise ValueError(f"inconsistent references for validation clip {key}")
    return groups


def _metrics(clips: Sequence[ScoredClip], threshold: float) -> dict:
    errors = ErrorCounts()
    voiced = rejected = instrumental = alarms = suppressed = 0
    for clip in clips:
        prediction = abstain(clip.prediction, clip.confidence, threshold)
        if clip.reference:
            voiced += 1
            rejected += not prediction
            suppressed += bool(clip.prediction and not prediction)
            errors.add(clip.reference, prediction)
        else:
            instrumental += 1
            alarms += bool(prediction)
    return {
        "clips": len(clips),
        "voiced_clips": voiced,
        "voiced_rejects": rejected,
        "voiced_reject_rate": rejected / voiced if voiced else None,
        "voiced_gate_suppressions": suppressed,
        "instrumental_clips": instrumental,
        "instrumental_false_alarms": alarms,
        "instrumental_false_alarm_rate": alarms / instrumental if instrumental else None,
        **errors.report(),
    }


def _before_after(clips: Sequence[ScoredClip], threshold: float) -> dict:
    return {"before": _metrics(clips, 0.0), "after": _metrics(clips, threshold)}


def calibrate(checkpoint: Path, manifest: Path, max_false_alarm_rate: float, *, fixture: bool = False) -> dict:
    frozen = not fixture or (manifest.parent / "freeze.json").is_file()
    if frozen:
        from scripts.prepare_lyrics import verify_frozen

        if verify_frozen(manifest.parent).resolve() != manifest.resolve():
            raise ValueError("calibration requires the frozen preparation manifest")
    manifest_digest = file_sha256(manifest)
    checkpoint_digest = file_sha256(checkpoint)
    validation = [row for row in read_manifest(manifest) if row.split == "validation"]
    if not validation:
        raise ValueError("calibration requires explicitly labeled validation rows")
    groups = _group_validation(validation)
    recognizer = Recognizer.load(checkpoint)
    if frozen and recognizer.training_manifest_sha256 != manifest_digest:
        raise ValueError("checkpoint was not trained from this frozen manifest")
    recognizer.check_held_out(validation)

    scored: dict[str, dict[tuple[str, str], ScoredClip]] = {}
    for key, group in groups.items():
        if "vocal" not in group:
            continue
        for condition, example in group.items():
            prediction, confidence = decode_with_confidence(recognizer.posteriors(example.audio))
            scored.setdefault(condition, {})[key] = ScoredClip(example.text, prediction, confidence)
    instrumentals = list(scored.get("instrumental", {}).values())
    threshold = select_threshold([clip.confidence for clip in instrumentals], max_false_alarm_rate)
    conditions = {}
    for condition, clips in sorted(scored.items()):
        report = _before_after(list(clips.values()), threshold)
        if condition not in ("vocal", "instrumental"):
            report["matched_vocal"] = _before_after([scored["vocal"][key] for key in clips], threshold)
        conditions[condition] = report
    if file_sha256(checkpoint) != checkpoint_digest or file_sha256(manifest) != manifest_digest:
        raise ValueError("checkpoint or manifest changed during calibration")
    return {
        "version": CALIBRATION_VERSION,
        "confidence": CONFIDENCE_METHOD,
        "checkpoint_sha256": checkpoint_digest,
        "manifest_sha256": manifest_digest,
        "split": "validation",
        "fixture": fixture,
        "max_false_alarm_rate_exclusive": max_false_alarm_rate,
        "threshold": threshold,
        "validation_excerpts": len(groups),
        "validation_vocal_excerpts": len(scored["vocal"]),
        "paired_validation_excerpts": sum("vocal" in group and len(group) > 1 for group in groups.values()),
        "overall": _before_after([clip for condition in scored.values() for clip in condition.values()], threshold),
        "conditions": conditions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="checkpoint-bound JSON calibration")
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.05, help="exclusive validation limit")
    parser.add_argument("--fixture", action="store_true", help="allow an unfrozen synthetic fixture manifest")
    args = parser.parse_args(argv)
    temporary = None
    try:
        if args.output.resolve() in (args.manifest.resolve(), args.checkpoint.resolve()):
            raise ValueError("calibration output must differ from the manifest and checkpoint")
        report = calibrate(args.checkpoint, args.manifest, args.max_false_alarm_rate, fixture=args.fixture)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(report, file, indent=2)
            file.write("\n")
        os.replace(temporary, args.output)
    except (OSError, ValueError, TypeError) as error:
        print(f"abstention calibration: {error}", file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
