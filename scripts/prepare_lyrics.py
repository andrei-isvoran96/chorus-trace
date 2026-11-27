"""Build and verify a frozen lyric corpus from reviewed, permissioned WAV clips."""

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from chorustrace.data import (
    SPLITS,
    file_sha256,
    lyric_field,
    read_jsonl,
    split_field,
    string_field,
    write_jsonl,
)
from chorustrace.prepare import SourceClip, render_clips


def _read_rights(path: Path) -> dict:
    rights = {}
    for location, row in read_jsonl(path):
        rights_id = string_field(row, "rights_id", location)
        if rights_id in rights:
            raise ValueError(f"{location}: duplicate rights_id {rights_id!r}")
        for field in ("source", "recording_permission", "lyrics_permission"):
            string_field(row, field, location)
        for field in ("training_allowed", "checkpoint_release_allowed"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"{location}: {field} must be a JSON boolean")
        rights[rights_id] = row
    return rights


def _read_tracks(row: dict, directory: Path, location: str) -> tuple[Path | None, ...]:
    vocal, mix = (
        (directory / string_field(row, field, location)).resolve() if field in row else None
        for field in ("vocal", "mix")
    )
    if vocal is None and mix is None:
        raise ValueError(f"{location}: supply a vocal or original mix WAV")
    backing = row.get("accompaniment", [])
    if isinstance(backing, str):
        backing = [backing]
    if not isinstance(backing, list) or any(not isinstance(path, str) or not path.strip() for path in backing):
        raise ValueError(f"{location}: accompaniment must be a WAV path or list of WAV paths")
    parts = tuple((directory / path).resolve() for path in backing)
    if len(set(parts)) != len(parts) or vocal in parts or mix in parts:
        raise ValueError(f"{location}: accompaniment tracks must be distinct")
    tracks = (vocal, mix, *parts)
    if sum(path is not None for path in tracks) > 1 and row.get("alignment_verified") is not True:
        raise ValueError(f"{location}: mark aligned WAVs with alignment_verified=true after checking sync")
    return tracks


def _read_clips(path: Path, rights: dict, max_seconds: float) -> list[SourceClip]:
    clips = []
    keys = set()
    recordings = {}
    intervals = defaultdict(list)
    track_owners = {}
    for location, row in read_jsonl(path):
        song = string_field(row, "song_id", location).strip().casefold()
        clip = string_field(row, "clip_id", location).strip()
        if (song, clip) in keys:
            raise ValueError(f"{location}: duplicate clip {song!r} / {clip!r}")
        keys.add((song, clip))
        info = {
            field: string_field(row, field, location).strip()
            for field in ("artist", "genre", "recording_id", "rights_id")
        }
        info["artist"] = " ".join(info["artist"].casefold().split())
        info["recording_id"] = info["recording_id"].casefold()
        recording_id = info["recording_id"]
        permission = rights.get(info["rights_id"])
        if permission is None or permission["training_allowed"] is not True:
            raise ValueError(f"{location}: no affirmative training permission for {info['rights_id']!r}")
        if row.get("language") != "en" or row.get("annotation_verified") is not True:
            raise ValueError(f"{location}: supply reviewed single-lead English annotations")
        text = lyric_field(row, location, allow_empty=False)
        start, end = row.get("start"), row.get("end")
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end))
            or not 0 <= start < end
            or end - start > max_seconds
        ):
            raise ValueError(f"{location}: require increasing start/end seconds and at most {max_seconds:g}s per clip")
        bounds = float(start), float(end)
        tracks = _read_tracks(row, path.parent, location)
        identity = (song, info["artist"], info["genre"], info["rights_id"], tracks)
        if recordings.setdefault(recording_id, identity) != identity:
            raise ValueError(f"{location}: inconsistent metadata or source WAVs for recording {recording_id!r}")
        for track in tracks[:2]:
            if track is None:
                continue
            if track_owners.setdefault(track, song) != song:
                raise ValueError(f"{location}: WAV reused under another song_id: {track}")
        intervals[recording_id].append((start, end, location))
        split = split_field(row, location)
        clips.append(SourceClip(song, clip, split, text, tracks, bounds, dict(info, start=start, end=end)))

    if not clips:
        raise ValueError(f"{path}: source manifest is empty")
    for recording, spans in intervals.items():
        spans.sort()
        for previous, current in pairwise(spans):
            if current[0] < previous[1]:
                raise ValueError(f"{current[2]}: overlapping excerpts in recording {recording!r}")
    # Keep clips from the same recording adjacent so the renderer decodes its WAVs once.
    return sorted(clips, key=lambda clip: (clip.metadata["recording_id"], clip.bounds, clip.clip_id))


def _components(clips: list[SourceClip]) -> list[list[SourceClip]]:
    # Every cover connects its performer to the other artists who sing that composition.
    links = defaultdict(set)
    for clip in clips:
        artist = "artist", clip.metadata["artist"]
        song = "song", clip.song_id
        links[artist].add(song)
        links[song].add(artist)
    root_by_key = {}
    for key in sorted(links):
        pending = [key]
        while pending:
            connected = pending.pop()
            if connected not in root_by_key:
                root_by_key[connected] = key
                pending.extend(links[connected])
    groups = defaultdict(list)
    for clip in clips:
        groups[root_by_key["song", clip.song_id]].append(clip)
    return list(groups.values())


def _split_clips(clips: list[SourceClip], seed: int) -> list[SourceClip]:
    groups = _components(clips)
    explicit = {clip.split for clip in clips}
    if None not in explicit:
        if any(len({clip.split for clip in group}) > 1 for group in groups):
            raise ValueError("explicit splits must keep artists and all versions of each song together")
        return clips
    if len(explicit) > 1:
        raise ValueError("provide an explicit split on every row or omit it throughout")

    def duration(group):
        return sum(clip.bounds[1] - clip.bounds[0] for clip in group)

    random.Random(seed).shuffle(groups)
    groups.sort(key=duration, reverse=True)
    allowed = SPLITS[:len(groups)]
    total = sum(duration(group) for group in groups)
    targets = dict(zip(SPLITS, (0.8 * total, 0.1 * total, 0.1 * total)))
    used = dict.fromkeys(allowed, 0.0)
    assignments = {}
    for index, group in enumerate(groups):
        empty = [split for split in allowed if used[split] == 0]
        candidates = empty if len(groups) - index == len(empty) else allowed
        split = max(candidates, key=lambda name: targets[name] - used[name])
        used[split] += duration(group)
        assignments.update({clip.song_id: split for clip in group})
    return [replace(clip, split=assignments[clip.song_id]) for clip in clips]


def _audio_hashes(directory: Path, rows: list[dict]) -> dict[str, str]:
    hashes = {}
    split_by_hash = {}
    for row in rows:
        audio = row["audio"]
        path = Path(audio)
        if path.is_absolute() or ".." in path.parts or audio in hashes:
            raise ValueError(f"invalid or duplicate prepared audio path {audio!r}")
        digest = hashes[audio] = file_sha256(directory / path)
        split = row["split"]
        if split_by_hash.setdefault(digest, split) != split:
            raise ValueError(f"{audio}: identical WAV occurs in multiple splits")
    return hashes


def freeze(
    sources: str | Path,
    rights: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    max_seconds: float = 30.0,
) -> Path:
    """Prepare a new corpus directory and bind its manifest and WAVs to a lock."""
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    sources, rights, output_dir = Path(sources), Path(rights), Path(output_dir)
    source_digest, rights_digest = file_sha256(sources), file_sha256(rights)
    permissions = _read_rights(rights)
    clips = _split_clips(_read_clips(sources, permissions, max_seconds), seed)
    training_ids = sorted({clip.metadata["rights_id"] for clip in clips if clip.split == "train"})
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        batch = output_dir / "clips"
        batch.mkdir()
        manifest = output_dir / "manifest.jsonl"
        rows = render_clips(clips, batch, output_dir, check_alignment=True)
        write_jsonl(manifest, rows)
        lock = {
            "version": 1,
            "split_seed": seed,
            "max_clip_seconds": max_seconds,
            "sources_sha256": source_digest,
            "rights_sha256": rights_digest,
            "training_rights_ids": training_ids,
            "training_checkpoint_release_allowed": bool(training_ids) and all(
                permissions[identifier]["checkpoint_release_allowed"] for identifier in training_ids
            ),
            "manifest_sha256": file_sha256(manifest),
            "audio_sha256": _audio_hashes(output_dir, rows),
        }
        if file_sha256(sources) != source_digest or file_sha256(rights) != rights_digest:
            raise ValueError("source or rights manifest changed during preparation")
        temporary_lock = output_dir / ".freeze.json"
        temporary_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_lock.replace(output_dir / "freeze.json")
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return manifest


def verify_frozen(output_dir: str | Path) -> Path:
    """Verify frozen metadata and audio before fitting, calibration, or scoring."""
    directory = Path(output_dir)
    lock = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise TypeError(f"{directory / 'freeze.json'}: expected JSON object")
    manifest = directory / "manifest.jsonl"
    if lock.get("version") != 1 or file_sha256(manifest) != lock.get("manifest_sha256"):
        raise ValueError(f"{manifest}: manifest does not match freeze.json")
    rows = [row for _, row in read_jsonl(manifest)]
    if _audio_hashes(directory, rows) != lock.get("audio_sha256"):
        raise ValueError(f"{directory}: prepared audio does not match freeze.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze", help="render WAV excerpts into a new, frozen dataset")
    prepare.add_argument("sources", type=Path)
    prepare.add_argument("rights", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--max-seconds", type=float, default=30.0)
    verify = commands.add_parser("verify", help="verify an existing frozen dataset")
    verify.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            print(freeze(args.sources, args.rights, args.output, args.seed, args.max_seconds))
        else:
            print(verify_frozen(args.output))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
