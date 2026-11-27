"""Stage published instrumental-solo ranges for development-only diagnostics."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from benchmarks.screen_clip1_development import _identity, _performer_ids
from chorustrace.data import read_jsonl, string_field, write_jsonl
from chorustrace.fulltrack_data import FullTrack, _audio_signature, _instrumental_intervals, _read_lock


METHOD = "https://staff.aist.go.jp/m.goto/PAPER/ISMIR2011mauch.pdf"
MARGIN_SECONDS = 0.25
MIN_SOLO_SECONDS = 1.0
MAX_END_DELTA_SECONDS = 0.05
_SOLO_LABEL = re.compile(r"s:[a-z]+", re.IGNORECASE)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _solo_intervals(annotation: str, duration: float) -> tuple[list[list[float]], float]:
    events = []
    for line in annotation.splitlines():
        if not line.strip():
            continue
        time, label = line.split()
        start = float(time)
        if not math.isfinite(start) or start < 0 or (events and start < events[-1][0]):
            raise ValueError("invalid RWC activity timestamp order")
        events.append((start, label))
    end = events[-1][0] if events else 0.0
    end_delta = end - duration
    if abs(end_delta) > MAX_END_DELTA_SECONDS:
        raise ValueError("RWC activity endpoint does not match the staged WAV duration")

    intervals = [
        [start + MARGIN_SECONDS, stop - MARGIN_SECONDS]
        for (start, label), (stop, _) in zip(events, events[1:])
        if _SOLO_LABEL.fullmatch(label) and stop - start > MIN_SOLO_SECONDS
    ]
    if intervals and events[-1][1] != "e":
        raise ValueError("RWC activity file with eligible solos has no terminal event")
    return intervals, end_delta


def prepare(source: Path, development: Path, lock: Path, output: Path) -> Path:
    """Read the lock fingerprints and stage disjoint RWC annotation evidence."""
    frozen = _read_lock(lock)
    existing = [row for _, row in read_jsonl(development)]
    identities = {
        _identity(row[field].split(":")[-1])
        for row in [*frozen, *existing]
        for field in ("song_id", "lyric_id", "recording_family_id")
    }
    identities.update(_identity(row["source_title"].split("(", 1)[0]) for row in existing if row.get("source_title"))
    performers = {person for row in [*frozen, *existing] for artist in row["artist_ids"]
                  for person in _performer_ids(artist)}
    audio_hashes = {row["audio_sha256"] for row in [*frozen, *existing] if row.get("audio_sha256")}
    pcm_hashes = {row["pcm_sha256"] for row in [*frozen, *existing] if row.get("pcm_sha256")}

    inventory = json.loads((source / "inventory.json").read_bytes())
    if inventory["license"] != "CC BY-NC 4.0":
        raise ValueError("expected RWC noncommercial research audio release")
    annotation_root = source / "published_drafts"
    annotations = json.loads((annotation_root / "index.json").read_bytes())
    if annotations["license"] != "https://creativecommons.org/licenses/by-nc/4.0/":
        raise ValueError("expected RWC noncommercial annotation archive")
    by_id = {entry["track_id"]: entry for entry in inventory["tracks"]}
    prepared = []
    for entry in annotations["tracks"]:
        activity = entry["vocal_activity"]
        if activity is None:
            continue
        track_id = entry["track_id"]
        track = by_id[track_id]
        if not re.fullmatch(r"RWC_P\d{3}", track_id) or track["condition"] != "original_mix":
            raise ValueError(f"{track_id}: expected a published original RWC Popular mix")
        subtitle = _identity(track["title"])
        artists = _performer_ids(track["artist"])
        if subtitle in identities or _identity(track_id) in identities or artists & performers:
            continue
        member = f"AIST_RWC-MDB-P-2001_VOCA_INST/RM-P{track_id[-3:]}.VOCA_INST.TXT"
        if not activity["member"].endswith(f"/{member}"):
            raise ValueError(f"{track_id}: RWC annotation archive entry does not match the recording")
        raw = (annotation_root / activity["path"]).read_bytes()
        if _sha256(raw) != activity["source_sha256"]:
            raise ValueError(f"{track_id}: unexpected RWC instrumental annotation hash")
        intervals, end_delta = _solo_intervals(raw.decode("utf-8-sig"), track["duration_seconds"])
        if not intervals:
            continue
        audio = source / track["audio"]
        size = {"source_frames": track["frames"], "source_sample_rate": track["sample_rate"]}
        duration, audio_hash, pcm_hash = _audio_signature(audio, size, track_id)
        if audio_hash != track["sha256"] or duration != track["duration_seconds"]:
            raise ValueError(f"{track_id}: RWC audio does not match the staged release inventory")
        if audio_hash in audio_hashes or pcm_hash in pcm_hashes:
            raise ValueError(f"{track_id}: RWC audio overlaps another split")
        audio_hashes.add(audio_hash)
        pcm_hashes.add(pcm_hash)

        reference = {
            "checked_by": "Mauch, Fujihara, Yoshii and Goto published manual activity annotations",
            "review_evidence": f"{METHOD} section 4.1; {annotations['source']}/blob/main/{member}",
            "instrumental_intervals": intervals,
            "source_activity_archive_sha256": annotations["archive_sha256"],
            "source_activity_file_sha256": activity["source_sha256"],
            "source_activity_license": annotations["license"],
            "annotation_end_minus_wav_seconds": end_delta,
            "margin_seconds": MARGIN_SECONDS,
            "minimum_source_solo_seconds_exclusive": MIN_SOLO_SECONDS,
        }
        _instrumental_intervals(reference, (), duration, track_id)
        reference_bytes = (json.dumps(reference, indent=2) + "\n").encode()
        row = size | {
            "track_id": track_id, "song_id": subtitle, "lyric_id": subtitle,
            "recording_family_id": track_id, "artist_ids": sorted(artists), "genre": track["genre"],
            "split": "validation", "full_track": True, "condition": "original_mix", "diagnostic_only": True,
            "audio": os.path.relpath(audio, output), "reference": f"{track_id}.json",
            "audio_source": track["archive_url"], "recording_permission": inventory["recording_permission"],
            "audio_sha256": audio_hash, "pcm_sha256": pcm_hash, "reference_sha256": _sha256(reference_bytes),
        }
        prepared.append((row, reference_bytes))

    if not prepared:
        raise ValueError("no disjoint RWC instrumental diagnostics")
    output.mkdir(mode=0o700, parents=True)
    for row, reference in prepared:
        (output / row["reference"]).write_bytes(reference)
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, [row for row, _ in prepared])
    return manifest


def read_bank(manifest: Path) -> list[FullTrack]:
    """Load only the separate development diagnostic bank for explicit scoring."""
    tracks = []
    for location, row in read_jsonl(manifest):
        if row.get("split") != "validation" or row.get("diagnostic_only") is not True:
            raise ValueError(f"{location}: expected a development-only diagnostic")
        audio = manifest.parent / string_field(row, "audio", location)
        reference_path = manifest.parent / string_field(row, "reference", location)
        duration, wav, pcm = _audio_signature(audio, row, location)
        reference_source = reference_path.read_bytes()
        if (wav, pcm, _sha256(reference_source)) != (row["audio_sha256"], row["pcm_sha256"], row["reference_sha256"]):
            raise ValueError(f"{location}: development diagnostic no longer matches inventory hashes")
        intervals = _instrumental_intervals(json.loads(reference_source), (), duration, str(reference_path))
        if not intervals:
            raise ValueError(f"{reference_path}: expected instrumental activity intervals")
        tracks.append(FullTrack(
            track_id=row["track_id"], song_id=row["song_id"], artist_ids=tuple(row["artist_ids"]),
            lyric_id=row["lyric_id"], recording_family_id=row["recording_family_id"], genre=row["genre"],
            split="validation", audio=audio, text="", duration_seconds=duration,
            instrumental_intervals=intervals, diagnostic_only=True,
        ))
    if not tracks:
        raise ValueError(f"{manifest}: no development diagnostics")
    return tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("development", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(prepare(args.source, args.development, args.lock, args.output))
