"""Build training-only CTC targets from CLIP1's published pronunciations."""

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.prepare_clip1_training import (
    METADATA_MEMBER, METADATA_MEMBER_MD5, RELEASE, _file_hexdigest, _md5, _verify_file,
)
from chorustrace.audio import FeatureConfig
from chorustrace.ctc import minimum_frames
from chorustrace.data import normalize_text, read_jsonl, split_field
from chorustrace.fulltrack_data import _check_splits, _ids, _load_track, _read_lock


ANNOTATION_METHOD = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12861282/"
_SIGNAL = re.compile(r"clip1:([0-9a-f]{24})")
_PHONE = re.compile(r"[a-zA-Z]{1,2}")
_GRIDS = (FeatureConfig(hop_size=160), FeatureConfig(hop_size=320), FeatureConfig(hop_size=400),
          FeatureConfig(hop_size=480, window_size=512))


def _published_labels(path: Path, signals: set[str]) -> tuple[dict[str, dict], str]:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(METADATA_MEMBER)
        if not member.isfile():
            raise ValueError(f"{path}: expected a regular official TRAIN metadata member")
        with archive.extractfile(member) as stream:
            payload = stream.read()
    checksum = _md5(payload).hexdigest()
    if checksum != METADATA_MEMBER_MD5:
        raise ValueError(f"{path}: training metadata member failed its published checksum")
    labels = {}
    for row in json.loads(payload):
        signal = row["signal"]
        if signal not in signals:
            continue
        if signal in labels:
            raise ValueError(f"CLIP1 {signal}: duplicate or alternative published pronunciations")
        labels[signal] = row
    if labels.keys() != signals:
        raise ValueError(f"{path}: {len(signals - labels.keys())} staged clips lack published TRAIN annotations")
    return labels, checksum


def _phonemes(row: dict, signal: str) -> tuple[str, ...]:
    pronunciation = row.get("prompt_pronunciation")
    prompt = row.get("prompt")
    if not isinstance(pronunciation, str) or not isinstance(prompt, str):
        raise ValueError(f"CLIP1 {signal}: missing normalized prompt or pronunciation")
    groups = [group.split("-") for group in pronunciation.split(" ")]
    phones = tuple(phone for group in groups for phone in group)
    if (
        any(not _PHONE.fullmatch(phone) for phone in phones)
        or len(groups) != len(prompt.split()) or len(groups) != row.get("n_words")
        or len(phones) != row.get("n_phonemes")
    ):
        raise ValueError(f"CLIP1 {signal}: malformed phoneme groups, alternative pronunciations, or count mismatch")
    return phones


def _train_clips(manifest: Path, lock: Path) -> dict:
    signatures = [("test", entry) for entry in _read_lock(lock)]
    training = {}
    for location, row in read_jsonl(manifest):
        split = split_field(row, location, required=True)
        if split == "test":
            continue
        if split == "validation":
            # Only declared identities are needed for CLIP1 split isolation.
            signatures.append((split, _ids(row, location)))
            continue
        loaded = _load_track(manifest, location, row, split)
        signatures.append((split, loaded.signature))
        match = _SIGNAL.fullmatch(loaded.track.track_id)
        if match:
            if row.get("audio_sha256") != loaded.signature["audio_sha256"]:
                raise ValueError(f"{location}: WAV differs from the verified CLIP1 staging checksum")
            training[match.group(1)] = row, loaded.track
    _check_splits(signatures)
    if not training:
        raise ValueError(f"{manifest}: no staged CLIP1 training clips")
    return training


def _frames(row: dict, grid: FeatureConfig) -> int:
    rate = row["source_sample_rate"]
    frames = (row["source_frames"] * grid.sample_rate + rate // 2) // rate
    remaining = max(0, frames - grid.window_size)
    return 1 + (remaining + grid.hop_size - 1) // grid.hop_size


def prepare(manifest: Path, lock: Path, metadata: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    metadata_md5 = _verify_file("metadata_archive", metadata)
    clips = _train_clips(manifest, lock)
    labels, member_md5 = _published_labels(metadata, set(clips))
    examples = {}
    inventory: Counter = Counter()
    case_variant_examples = changed_prompt_examples = 0
    for signal, (row, track) in sorted(clips.items()):
        source = labels[signal]
        fma = int(source["fma"])
        if (
            source.get("hearing_loss") != "No Loss" or track.full_track or track.condition != "original_mix"
            or track.recording_family_id != f"fma:{fma}"
            or not row["audio_source"].endswith(f"/train/unprocessed/{signal}_unproc.flac")
        ):
            raise ValueError(f"CLIP1 {signal}: expected the published unprocessed training source")
        reference = json.loads((manifest.parent / row["reference"]).read_bytes())
        if reference.get("text") != source.get("original_prompt") or track.text != normalize_text(source["original_prompt"]):
            raise ValueError(f"CLIP1 {signal}: staged reference differs from the published original_prompt")
        phones = _phonemes(source, signal)
        inventory.update(phones)
        examples[track.track_id] = phones, hashlib.sha256(track.text.encode("utf-8")).hexdigest(), row
        case_variant_examples += any(phone != phone.lower() for phone in phones)
        changed_prompt_examples += normalize_text(source["prompt"]) != track.text

    vocabulary = sorted(inventory)
    indices = {phone: index for index, phone in enumerate(vocabulary, start=1)}
    targets = {"version": 1, "vocabulary": vocabulary, "examples": {}}
    margins = {str(grid.hop_size): [] for grid in _GRIDS}
    for track_id, (phones, digest, row) in examples.items():
        encoded = [indices[phone] for phone in phones]
        targets["examples"][track_id] = {"grapheme_sha256": digest, "phones": encoded}
        required = minimum_frames(encoded)
        for grid in _GRIDS:
            margins[str(grid.hop_size)].append(_frames(row, grid) - required)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "targets"
        staged.mkdir()
        sidecar = staged / "phoneme-targets.json"
        sidecar.write_text(json.dumps(targets, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        report = {
            "split": "train", "examples": len(examples), "source_release": RELEASE,
            "annotation_method": ANNOTATION_METHOD, "metadata_archive_md5": metadata_md5,
            "metadata_member": METADATA_MEMBER, "metadata_member_md5": member_md5,
            "source_development_manifest_sha256": _file_hexdigest(manifest, hashlib.sha256()),
            "frozen_test_lock_sha256": _file_hexdigest(lock, hashlib.sha256()),
            "sidecar_sha256": _file_hexdigest(sidecar, hashlib.sha256()),
            "exact_prompt_reference_matches": len(examples), "duplicate_or_alternative_label_rows": 0,
            "malformed_pronunciation_rows": 0, "case_variant_examples": case_variant_examples,
            "different_published_normalized_prompt_examples": changed_prompt_examples,
            "scope": "clip-level BEEP dictionary targets for the published normalized prompt, preserving token case",
            "blank_id": 0, "phone_vocabulary_size": len(vocabulary),
            "casefolded_phone_vocabulary_size": len({phone.casefold() for phone in vocabulary}),
            "phone_token_counts": inventory,
            "ctc": {str(grid.hop_size): {
                "sample_rate": grid.sample_rate, "window_samples": grid.window_size,
                "feasible": sum(margin >= 0 for margin in margins[str(grid.hop_size)]),
                "minimum_spare_frames": min(margins[str(grid.hop_size)]),
            } for grid in _GRIDS},
        }
        (staged / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staged, output)
    return output / "phoneme-targets.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "lock", "metadata", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(prepare(args.manifest, args.lock, args.metadata, args.output))
