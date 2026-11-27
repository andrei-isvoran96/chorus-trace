"""Reconstruct the existing local-only Jam-ALT and RWC frozen benchmark."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile

from benchmarks.prepare_jam_alt_holdout import prepare
from benchmarks.prepare_rwc_test_candidates import _check_copy, _copy_range
from chorustrace.data import read_jsonl
from chorustrace.fulltrack_data import _check_loaded, _load_selected, _test_signatures, verify_fulltrack_test


_HF = "https://huggingface.co"
_REVIEW = "jamendolyrics/jam-alt"
_REVIEW_REV = "a431ec53fd0d54547f2891789c643a6288df9dae"
_SOURCE = "jamendolyrics/jamendolyrics"
_SOURCE_REV = "e9a8a63e7c828a0a7152c46752de5c24ae65de94"
_ENGLISH = "subsets/en/metadata.jsonl"
_REVIEW_SHA = "3de8cd250d7ec05dbb5126f553a1b0d1b97c88ece68b25e293fa17bf064f4175"
_SOURCE_SHA = "daedfe6f5f7445276f4742ebdd633816ecae7ab810baf853e8216ab03d89d705"
_CACHED_REVIEW_SHA = "16b6a4922f5d3f3c0951b63297b883f4f525bce1abb964ff957b521bf38d21d6"
_CACHED_SOURCE_SHA = "a214dbc7ca88fee4e41effa559502777365f5798e28ef585812e8413330be73b"
_LICENSE_SHA = "813dce6d4ec5987ec6aa8ff9548f18a6b6f43b1c7c362cbb4252e757123faa89"
_MP3_INDEX_SHA = "325805bc96a5bec8c1d12ee2d827e20e1321fff65a59dcc9f5293179a5ad5170"
_RWC_METADATA = "https://github.com/rwc-music/rwc-annotations/blob/main/metadata.csv"
_RWC_METADATA_PIN = (
    "https://raw.githubusercontent.com/rwc-music/rwc-annotations/7c2053edc3ffd26066660502430f33cdc18ad6bf/metadata.csv"
)
_RWC_METADATA_SHA = "eba3fe129f835db58a9c23dad554547201ab02ef3af90c10857ef5b50b8b99da"
_RWC_ZIP = "https://zenodo.org/records/18656623/files/RWC-J.zip?download=1"
_RWC_BYTES = 2_127_754_512
_RWC_TAIL_SHA = "21e149edd024d559a156176a54a68a4a3add69a67bd622fdac8300429204e71e"
_RWC_TRACKS = {
    "RWC_J001": "6a92ea17863d0b90914dff827f2d8b437963b5ef2f0b951b8a62158cd022b122",
    "RWC_J013": "2c5d4b33da608667d15cd2ecf5c8eddb727437b035d88a66865e00a9451ef2ff",
    "RWC_J007": "1d6d1948c846f0ecaf0cc525fce5ded889fddd1cf32d3d711b5ee321ca2a18ed",
    "RWC_J025": "8be9f93a88c71ddee45858bd61d78d46a3f19990b334757ac295f74c5b5b336e",
    "RWC_J019": "0a3eb8f94f4138b21f18f07483cdab0a0bd290d6e4b1715173fda0c4769f40c3",
}
_FROZEN_AT = "2026-11-27T09:10:47.031421+00:00"
_MANIFEST_SHA = "40c5579ce680757c5dc7706a3bec482066f079736392c18bc912feb79231b39a"
_LOCK_SHA = "9fa1241d905db1d31b1f1695303bb1cecb8c59f1ed1d93aad15cb1fb62ec22c2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fetch(url: str):
    return urlopen(Request(url, headers={"User-Agent": "ChorusTrace frozen benchmark acquisition"}), timeout=120)


def _stage(url: str, target: Path, cache: Path | None, sha: str, *, size: int | None = None) -> None:
    with (cache.open("rb") if cache is not None else _fetch(url)) as source, target.open("xb") as destination:
        digest = hashlib.sha256()
        count = 0
        for block in iter(lambda: source.read(1 << 20), b""):
            destination.write(block)
            digest.update(block)
            count += len(block)
    if digest.hexdigest() != sha or (size is not None and count != size):
        raise ValueError(f"{target.name}: pinned source checksum or size differs")


def _jam_urls(dataset: str, revision: str, directory: str) -> tuple[str, str]:
    return (f"{_HF}/datasets/{dataset}/resolve/{revision}",
            f"{_HF}/api/datasets/{dataset}/tree/{revision}/subsets/en/{directory}")


def _audio_index(url: str, cache: Path | None) -> dict[str, tuple[str, int]]:
    with cache.open("rb") if cache is not None else _fetch(url) as source:
        rows = json.load(source)
    entries = sorted((Path(row["path"]).name, row["lfs"]["oid"], row["size"]) for row in rows)
    encoded = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()
    if len(entries) != 20 or _sha256(encoded) != _MP3_INDEX_SHA:
        raise ValueError("English MP3 index differs from the recorded frozen editions")
    return {name: (sha, size) for name, sha, size in entries}


def _wave_details(path: Path) -> dict:
    with path.open("rb") as stream, wave.open(stream, "rb") as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"frames": frames, "sample_rate": rate, "duration_seconds": frames / rate, "sha256": digest.hexdigest()}


def _acquire_jam(output: Path, cache: Path | None, ffmpeg: str) -> None:
    (output / "mixes").mkdir(parents=True)
    review, review_index = _jam_urls(_REVIEW, _REVIEW_REV, "audio")
    source, source_index = _jam_urls(_SOURCE, _SOURCE_REV, "mp3")
    revised = output / "metadata.jsonl"
    original = output / "original-metadata.jsonl"
    _stage(f"{review}/{_ENGLISH}", revised, cache / revised.name if cache else None,
           _CACHED_REVIEW_SHA if cache else _REVIEW_SHA)
    _stage(f"{source}/{_ENGLISH}", original, cache / original.name if cache else None,
           _CACHED_SOURCE_SHA if cache else _SOURCE_SHA)
    _stage(f"{source}/LICENSE", output / "original-LICENSE", cache / "original-LICENSE" if cache else None,
           _LICENSE_SHA)
    mp3_index = _audio_index(source_index, cache / "original-en-audio-tree.json" if cache else None)
    review_mp3 = _audio_index(review_index, cache / "revised-en-audio-tree.json" if cache else None)
    if mp3_index != review_mp3:
        raise ValueError("original and reviewed English MP3 editions differ")

    published = {row["name"]: row for _, row in read_jsonl(original) if row["language"] == "en"}
    tracks = []
    for _, row in read_jsonl(revised):
        if row["language"] != "en":
            continue
        track_id = _sha256(row["name"].encode())[:16]
        record = published[row["name"]]
        file_name = Path(record["file_name"]).name
        license_name = f"CC {record['license_type'].removeprefix('CC ')}"
        if file_name != Path(row["file_name"]).name or row["license_type"] != license_name:
            raise ValueError("reviewed metadata no longer identifies the licensed source mix")
        mp3_sha, mp3_size = mp3_index[file_name]
        source_url = f"{source}/subsets/en/mp3/{quote(file_name)}"
        audio = output / "mixes" / f"{track_id}.wav"
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            mp3 = Path(temporary) / f"{track_id}.mp3"
            _stage(source_url, mp3, cache / "mixes" / mp3.name if cache else None, mp3_sha, size=mp3_size)
            subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(mp3), "-map", "0:a:0",
                            "-c:a", "pcm_s16le", str(audio)], check=True)
        tracks.append(_wave_details(audio) | {
            "track_id": f"jamalt:{track_id}", "artist": row["artist"], "title": row["title"], "genre": row["genre"],
            "recording_license": license_name, "source_url": source_url,
            "published_mp3_sha256": mp3_sha, "source_size": mp3_size, "audio": audio.relative_to(output).as_posix(),
        })
    _write_json(output / "inventory.json", {
        "original_recording_license": f"{_HF}/datasets/{_SOURCE}/blob/{_SOURCE_REV}/LICENSE", "tracks": tracks,
    })


def _extract_rwc(output: Path) -> None:
    start = _RWC_BYTES - 65536
    with tempfile.TemporaryFile(dir=output) as sparse:
        _copy_range(_RWC_ZIP, _RWC_BYTES, start, _RWC_BYTES, sparse)
        sparse.seek(start)
        if _sha256(sparse.read()) != _RWC_TAIL_SHA:
            raise ValueError("RWC Jazz ZIP directory differs from frozen release")
        with ZipFile(sparse) as archive:
            offsets = sorted(entry.header_offset for entry in archive.infolist())
            stops = dict(zip(offsets, offsets[1:] + [archive.start_dir], strict=True))
            for track, expected in _RWC_TRACKS.items():
                entry = archive.getinfo(f"RWC-J/{track}.wav")
                _copy_range(_RWC_ZIP, _RWC_BYTES, entry.header_offset, stops[entry.header_offset], sparse)
                with archive.open(entry) as source, (output / f"{track}.wav").open("xb") as destination:
                    if _check_copy(source, entry, destination) != expected:
                        raise ValueError(f"{track}: RWC source WAV checksum differs")


def _acquire_rwc(output: Path, cache: Path | None) -> None:
    output.mkdir(parents=True)
    metadata = output / "metadata.csv"
    _stage(_RWC_METADATA_PIN, metadata, cache / "rwc/metadata.csv" if cache else None, _RWC_METADATA_SHA)
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        catalog = {row["RWCID"]: row for row in csv.DictReader(stream, delimiter=";") if row["CollID"] == "J"}
    if cache is None:
        _extract_rwc(output)
    tracks = []
    for track, expected in _RWC_TRACKS.items():
        row = catalog[track]
        if row["SingerInformation"] or row["SingingLanguage"] or not row["LiveInstruments"]:
            raise ValueError(f"{track}: RWC catalogue no longer supports an instrumental reference")
        audio = output / f"{track}.wav"
        if cache is not None:
            _stage(_RWC_ZIP, audio, cache / "rwc_jazz_diagnostics" / audio.name, expected)
        details = _wave_details(audio)
        if details["sha256"] != expected or abs(details["duration_seconds"] - float(row["duration"])) > 1 / details["sample_rate"]:
            raise ValueError(f"{track}: RWC WAV differs from the published complete recording")
        tracks.append(details | {
            "track_id": track, "audio": audio.name, "title": row["Title"], "artist": row["Artist"],
            "instrumental_intervals": [[0, details["duration_seconds"]]],
        })
    _write_json(output / "inventory.json", {"source": _RWC_ZIP, "metadata_source": _RWC_METADATA, "tracks": tracks})


def _restore_lock(manifest: Path) -> Path:
    if _sha256(manifest.read_bytes()) != _MANIFEST_SHA:
        raise ValueError("reconstructed manifest differs from original; verify source and FFmpeg 8.1.1 WAV durations")
    entries = _load_selected(manifest, ("test",))
    _check_loaded(entries)
    payload = {"version": 1, "frozen_at": _FROZEN_AT, "tracks": _test_signatures(entries)}
    lock_bytes = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if _sha256(lock_bytes) != _LOCK_SHA:
        raise ValueError("reconstructed lock differs from original; verify exact source and FFmpeg 8.1.1 WAV output")
    lock = manifest.with_name("test.lock.json")
    lock.write_bytes(lock_bytes)
    verify_fulltrack_test(manifest, lock)
    return lock


def acquire(destination: Path, *, source_cache: Path | None = None, ffmpeg: str = "ffmpeg") -> Path:
    """Create a new data tree containing the byte-identical original manifest and lock."""
    if destination.exists():
        raise FileExistsError(f"destination must be a new directory: {destination}")
    if shutil.which(ffmpeg) is None:
        raise ValueError("FFmpeg is required to reproduce the pinned WAVs; original version was 8.1.1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        data = Path(temporary) / "data"
        data.mkdir(mode=0o700)
        candidates = data / "public_test_candidates"
        jam, rwc = candidates / "jam_alt_audit", candidates / "rwc_jazz_diagnostics"
        _acquire_jam(jam, source_cache / "jam_alt_audit" if source_cache else None, ffmpeg)
        _acquire_rwc(rwc, source_cache)
        manifest = prepare(jam, rwc, data / "frozen_fulltrack")
        lock = _restore_lock(manifest)
        if destination.exists():
            raise FileExistsError(f"destination appeared during reconstruction: {destination}")
        os.rename(data, destination)
        return destination / lock.relative_to(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new local data directory")
    parser.add_argument("--source-cache", type=Path, help="optional original staged public_test_candidates directory")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg program (original version 8.1.1)")
    arguments = parser.parse_args()
    try:
        print(acquire(arguments.destination, source_cache=arguments.source_cache, ffmpeg=arguments.ffmpeg))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
