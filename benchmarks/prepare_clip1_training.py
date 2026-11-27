"""Stage audited English CLIP1 development excerpts as lossless PCM WAVs."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import wave
from pathlib import Path

from chorustrace.data import normalize_text, read_jsonl, string_field, write_jsonl
from chorustrace.fulltrack_data import read_fulltrack_manifest


RELEASE = "https://zenodo.org/records/17950664"
LICENSE_NAME = "clip1_track_license_1.00.csv"
METADATA_NAME = "cadenza_clip1_data.metadata.v1.3.tar.gz"
METADATA_MEMBER = "cadenza_data/metadata/train_metadata_with_phoneme.json"
METADATA_MEMBER_MD5 = "e8246d5b3e7bfd6f5fd58202465486b9"
VALIDATION_METADATA_MEMBER = "cadenza_data/metadata/valid_metadata_with_phoneme.json"
VALIDATION_METADATA_MEMBER_MD5 = "a945d7e87a77e72c6a69daf926e0cf42"
RELEASE_MD5S = {
    "training_archive": "a31b05e2e35198206eb81ad2121b42ae",
    "validation_archive": "19398629d6110ac02ce8b1e8ebfeddb7",
    "metadata_archive": "4851356e2b8ef5c156a46625a92c266d",
    "license_catalog": "6cb6a631ce446d3e7cc6ed7da305fdd6",
}
ALLOWED_LICENSES = frozenset((
    "Attribution", "Creative Commons Attribution", "Attribution 3.0 United States",
    "Attribution 3.0 International", "Attribution-ShareAlike",
    "Attribution-ShareAlike 3.0 International", "CC0 1.0 Universal",
))
NONCOMMERCIAL_LICENSES = frozenset((
    "Attribution-NonCommercial",
    "Attribution-NonCommercial 2.5",
    "Attribution-NonCommercial 3.0 International",
    "Attribution-NonCommercial-ShareAlike",
    "Attribution-NonCommercial-ShareAlike 3.0 (Will Not Appear in Vimeo Music Store)",
    "Attribution-NonCommercial-ShareAlike 3.0 International",
    "Attribution-Noncommercial 2.5 Switzerland",
    "Attribution-Noncommercial 3.0 Netherlands",
    "Attribution-Noncommercial 3.0 United States",
    "Attribution-Noncommercial-Share Alike 2.0 Germany",
    "Attribution-Noncommercial-Share Alike 2.0 UK: England",
    "Attribution-Noncommercial-Share Alike 2.5 Italy",
    "Attribution-Noncommercial-Share Alike 3.0 Germany",
    "Attribution-Noncommercial-Share Alike 3.0 Netherlands",
    "Attribution-Noncommercial-Share Alike 3.0 New Zealand",
    "Attribution-Noncommercial-Share Alike 3.0 United States",
))
BLOCK_SIZE = 1 << 20


def is_allowed_license(label: str | None, *, allow_noncommercial: bool = False) -> bool:
    """Match the CLIP1 source catalog's exact redistribution and adaptation terms."""
    if not isinstance(label, str):
        return False
    label = label.strip()
    return label in ALLOWED_LICENSES or (allow_noncommercial and label in NONCOMMERCIAL_LICENSES)


def _md5(data=b""):
    return hashlib.md5(data, usedforsecurity=False)


def _file_hexdigest(path: Path, digest) -> str:
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(label: str, path: Path) -> str:
    checksum = _file_hexdigest(path, _md5())
    if checksum != RELEASE_MD5S[label]:
        raise ValueError(f"{path}: MD5 differs from the published CLIP1 release")
    return checksum


def _selected_rows(
    shortlist: Path, metadata: Path, licenses: Path, split: str = "train", *, allow_noncommercial: bool = False,
) -> dict[str, dict]:
    selected = {}
    with licenses.open(encoding="utf-8-sig", newline="") as stream:
        published = {row["fma_track_id"]: row["licence"].strip() for row in csv.DictReader(stream)}
    for location, row in read_jsonl(shortlist):
        signal = string_field(row, "signal", location)
        if not re.fullmatch(r"[a-f0-9]{24}", signal) or signal in selected:
            raise ValueError(f"{location}: invalid or duplicate source signal")
        fma = row.get("fma")
        if type(fma) is not int or fma <= 0:
            raise ValueError(f"{location}: invalid FMA track ID")
        license_label = published.get(str(fma))
        audited = row.get("clip1_license"), row.get("track.license")
        if not is_allowed_license(license_label, allow_noncommercial=allow_noncommercial) or any(
            not isinstance(label, str) or label.strip() != license_label for label in audited
        ):
            raise ValueError(f"{location}: shortlist and official license must agree on an approved source term")
        for field in ("artist.id", "artist.name", "track.title"):
            string_field(row, field, location)
        selected[signal] = row
    if not selected:
        raise ValueError(f"{shortlist}: no audited CLIP1 excerpts")

    member = METADATA_MEMBER if split == "train" else VALIDATION_METADATA_MEMBER
    expected_md5 = METADATA_MEMBER_MD5 if split == "train" else VALIDATION_METADATA_MEMBER_MD5
    with tarfile.open(metadata, "r:gz") as archive:
        entry = archive.getmember(member)
        if not entry.isfile():
            raise ValueError(f"{metadata}: expected a regular {split} metadata member")
        with archive.extractfile(entry) as stream:
            source = stream.read()
    if _md5(source).hexdigest() != expected_md5:
        raise ValueError(f"{metadata}: {split} metadata checksum differs from the release manifest")

    transcripts = {}
    excerpts = set()
    for row in json.loads(source):
        signal = row["signal"]
        if signal not in selected:
            continue
        text = string_field(row, "original_prompt", f"CLIP1 signal {signal}")
        words = normalize_text(text)
        fma = selected[signal]["fma"]
        source_id = string_field(row, "fma", f"CLIP1 signal {signal}")
        if (
            not re.fullmatch(r"[0-9]+", source_id) or int(source_id) != fma or not words
            or signal in transcripts or (fma, words) in excerpts
        ):
            raise ValueError(f"CLIP1 signal {signal}: inconsistent catalogue or repeated excerpt")
        transcripts[signal] = text
        excerpts.add((fma, words))
    if transcripts.keys() != selected.keys():
        missing = sorted(selected.keys() - transcripts.keys())
        raise ValueError(f"{metadata}: missing published original_prompt for {len(missing)} selected signals")
    return {signal: row | {"original_prompt": transcripts[signal]} for signal, row in selected.items()}


def _decode_flac(source: bytes, path: Path, ffmpeg: str, max_seconds: float) -> str:
    if len(source) < 42 or source[:4] != b"fLaC" or source[4] & 127 or source[5:8] != b"\x00\x00\x22":
        raise ValueError(f"{path.stem}: expected FLAC STREAMINFO")
    info = int.from_bytes(source[18:26], "big")
    rate, channels, bits = info >> 44, ((info >> 41) & 7) + 1, ((info >> 36) & 31) + 1
    frames = info & ((1 << 36) - 1)
    digest = source[26:42]
    formats = {16: "s16le", 24: "s24le", 32: "s32le"}
    if not rate or not frames or frames / rate > max_seconds or bits not in formats or not any(digest):
        raise ValueError(f"{path.stem}: invalid FLAC format, clip length, or missing PCM checksum")
    sample_format = formats[bits]
    pcm = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-err_detect", "crccheck+explode",
         "-f", "flac", "-i", "pipe:0", "-map", "0:a:0", "-c:a", f"pcm_{sample_format}",
         "-f", sample_format, "pipe:1"],
        input=source, stdout=subprocess.PIPE, check=True,
    ).stdout
    width = bits // 8
    if len(pcm) != frames * channels * width or _md5(pcm).digest() != digest:
        raise ValueError(f"{path.stem}: decoded PCM differs from the FLAC STREAMINFO checksum")
    with wave.open(str(path), "wb") as wav:
        wav.setparams((channels, width, rate, frames, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return digest.hex()


def _stage_members(
    archive: Path, output: Path, selected: dict, ffmpeg: str, max_seconds: float, split: str
) -> list[dict]:
    audio_dir, references = output / "audio", output / "references"
    audio_dir.mkdir()
    references.mkdir()
    source_split = "train" if split == "train" else "valid"
    release_file = f"{RELEASE}/files/cadenza_clip1_data.{source_split}.v1.2.tar.gz"
    members = {f"cadenza_data/{source_split}/unprocessed/{signal}_unproc.flac": signal for signal in selected}
    rows = {}
    with gzip.open(archive, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as package:
            for entry in package:
                signal = members.get(entry.name.removeprefix("./"))
                if signal is None:
                    continue
                if signal in rows or not entry.isfile() or not 0 < entry.size <= 64 * BLOCK_SIZE:
                    raise ValueError(f"{entry.name}: invalid or repeated selected FLAC archive member")
                with package.extractfile(entry) as stream:
                    source = stream.read()
                if len(source) != entry.size:
                    raise ValueError(f"{entry.name}: incomplete FLAC archive member")

                row = selected[signal]
                wav_path = audio_dir / f"{signal}.wav"
                pcm_md5 = _decode_flac(source, wav_path, ffmpeg, max_seconds)
                with wave.open(str(wav_path), "rb") as wav:
                    frames, rate = wav.getnframes(), wav.getframerate()
                reference = references / f"{signal}.json"
                reference.write_text(json.dumps({
                    "text": row["original_prompt"], "complete_lead_vocal": True,
                    "transcribed_by": "CLIP1 published original_prompt annotation",
                }, ensure_ascii=False) + "\n", encoding="utf-8")
                source_id = f"fma:{row['fma']}"
                permission = (
                    f"Official CLIP1 per-track terms for {source_id}, {row['clip1_license']}; "
                    f"FMA catalog {row['track.license']}; {RELEASE}/files/{LICENSE_NAME}"
                )
                if row["clip1_license"].strip() in NONCOMMERCIAL_LICENSES:
                    permission = f"Noncommercial research only. {permission}"
                rows[signal] = {
                    "track_id": f"clip1:{signal}", "song_id": source_id, "lyric_id": source_id,
                    "recording_family_id": source_id, "artist_ids": [f"fma:artist:{row['artist.id']}", row["artist.name"]],
                    "source_title": row["track.title"], "genre": row.get("track.genre_top", "").strip() or "unspecified",
                    "split": split, "condition": "original_mix", "language": "en", "full_track": False,
                    "audio": wav_path.relative_to(output).as_posix(), "reference": reference.relative_to(output).as_posix(),
                    "audio_source": f"{release_file} :: {entry.name}",
                    "source_frames": frames, "source_sample_rate": rate,
                    "recording_permission": permission,
                    "lyrics_permission": f"CLIP1 published original_prompt and the same FMA source terms, {permission}",
                    "source_flac_sha256": hashlib.sha256(source).hexdigest(),
                    "source_pcm_md5": pcm_md5,
                    "audio_sha256": _file_hexdigest(wav_path, hashlib.sha256()),
                }
        # A streaming tar reader can stop at tar EOF before gzip's trailer.
        while compressed.read(BLOCK_SIZE):
            pass
    if rows.keys() != selected.keys():
        missing = selected.keys() - rows.keys()
        raise ValueError(f"{archive}: missing {len(missing)} selected unprocessed FLAC members")
    return [rows[signal] for signal in selected]


def prepare(
    audit: Path, archive: Path, output: Path, *, ffmpeg: str = "ffmpeg", max_seconds: float = 12,
    split: str = "train", allow_noncommercial: bool = False,
) -> Path:
    if split not in ("train", "validation"):
        raise ValueError("CLIP1 staging supports train and validation splits")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("maximum clip length must be finite and positive")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    shortlist = audit / ("shortlist.jsonl" if split == "train" else "validation-shortlist.jsonl")
    metadata, licenses = audit / METADATA_NAME, audit / LICENSE_NAME
    checksums = {
        "metadata_archive": _verify_file("metadata_archive", metadata),
        "license_catalog": _verify_file("license_catalog", licenses),
    }
    selected = _selected_rows(shortlist, metadata, licenses, split, allow_noncommercial=allow_noncommercial)
    archive_label = "training_archive" if split == "train" else "validation_archive"
    checksums[archive_label] = _verify_file(archive_label, archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        rows = _stage_members(archive, staged, selected, ffmpeg, max_seconds, split)
        manifest = staged / f"{split}.jsonl"
        write_jsonl(manifest, rows)
        tracks = read_fulltrack_manifest(manifest)
        report = {
            "split": split, "release": RELEASE, "clips": len(tracks),
            "source_tracks": len({track.song_id for track in tracks}),
            "catalog_artists": len({row["artist.id"] for row in selected.values()}),
            "duration_seconds": sum(track.duration_seconds for track in tracks),
            "max_clip_seconds": max(track.duration_seconds for track in tracks),
            "release_md5": checksums, "verified_gzip_crc": True,
            "shortlist_sha256": _file_hexdigest(shortlist, hashlib.sha256()),
            "manifest_sha256": _file_hexdigest(manifest, hashlib.sha256()),
        }
        if allow_noncommercial:
            report["noncommercial_source_clips"] = sum(
                row["clip1_license"].strip() in NONCOMMERCIAL_LICENSES for row in selected.values()
            )
        (staged / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staged, output)
    return output / f"{split}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path, help="directory with the shortlist, metadata tar and license CSV")
    parser.add_argument("archive", type=Path, help="official CLIP1 train or valid v1.2 tar.gz")
    parser.add_argument("output", type=Path, help="new directory for the split manifest and excerpts")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--allow-noncommercial", action="store_true", help="include audited BY-NC and BY-NC-SA excerpts")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable used for acquisition conversion")
    parser.add_argument("--max-clip-seconds", type=float, default=12)
    args = parser.parse_args()
    print(prepare(
        args.audit, args.archive, args.output, ffmpeg=args.ffmpeg, max_seconds=args.max_clip_seconds, split=args.split,
        allow_noncommercial=args.allow_noncommercial,
    ))


if __name__ == "__main__":
    main()
