"""Stage complete RWC original mixes for noncommercial test reference review."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
import wave
import zlib
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


RELEASE = "https://zenodo.org/records/18656623"
METADATA = "https://raw.githubusercontent.com/rwc-music/rwc-annotations/main/metadata.csv"
LICENSE = "https://creativecommons.org/licenses/by-nc/4.0/"
ARCHIVE_SIZES = {"P": 4071840278, "G": 3933055195}
BLOCK_SIZE = 1 << 20
ZIP_TAIL_SIZE = 65536


def select_candidates(metadata: Path) -> list[dict]:
    """Select tracks by catalog metadata before opening any recording."""
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream, delimiter=";")
            if row["CollID"] in ARCHIVE_SIZES and row["SingingLanguage"] == "English"
        ]
    cohorts = [
        ("pop", 8, [row for row in rows if row["CollID"] == "P" and row["GenreSub"] == "Pop"]),
        ("rock", 6, [row for row in rows if row["CollID"] == "G" and row["GenreMain"] == "Rock"]),
        ("hip-hop", 3, [row for row in rows if row["CollID"] == "G" and row["GenreSub"] == "Rap / Hip-Hop"]),
        (None, 7, [
            row for row in rows if row["CollID"] == "G"
            and row["GenreMain"] not in ("Popular", "Rock", "A Cappella")
            and row["GenreSub"] != "Rap / Hip-Hop"
        ]),
    ]
    artists: Counter = Counter()
    other_families: Counter = Counter()
    other_genres = set()
    selected = []
    for genre, count, candidates in cohorts:
        for _ in range(count):
            eligible = [row for row in candidates if genre or row["GenreSub"] not in other_genres]
            if not eligible:
                raise ValueError(f"RWC metadata has too few English {genre or 'other genre'} tracks")
            row = min(eligible, key=lambda row: (artists[row["Artist"]], other_families[row["GenreMain"]], row["RWCID"]))
            candidates.remove(row)
            artists[row["Artist"]] += 1
            if genre is None:
                other_families[row["GenreMain"]] += 1
                other_genres.add(row["GenreSub"])
            selected.append(row | {"genre": genre or row["GenreSub"].casefold()})
    return selected


def _copy_range(url: str, size: int, start: int, stop: int, stream) -> None:
    """Populate one exact byte range of an otherwise sparse ZIP file."""
    if not 0 <= start < stop <= size:
        raise ValueError("invalid ZIP HTTP byte range")
    request = Request(url, headers={
        "Range": f"bytes={start}-{stop - 1}",
        "Accept-Encoding": "identity",
    })
    with urlopen(request, timeout=90) as response:
        content_range = f"bytes {start}-{stop - 1}/{size}"
        if response.status != 206 or response.headers.get("Content-Range") != content_range:
            raise ValueError(f"{url}: expected HTTP 206 and Content-Range {content_range}")
        stream.seek(start)
        remaining = stop - start
        while remaining:
            block = response.read(min(BLOCK_SIZE, remaining))
            if not block:
                raise ValueError(f"{url}: truncated byte range {content_range}")
            stream.write(block)
            remaining -= len(block)
        if response.read(1):
            raise ValueError(f"{url}: extra bytes after byte range {content_range}")
    stream.flush()


def _check_copy(source, entry, target=None) -> str:
    digest = hashlib.sha256()
    crc, length = 0, 0
    for block in iter(lambda: source.read(BLOCK_SIZE), b""):
        length += len(block)
        crc = zlib.crc32(block, crc)
        digest.update(block)
        if target is not None:
            target.write(block)
    if length != entry.file_size or crc != entry.CRC:
        raise ValueError(f"{entry.filename}: bytes differ from the release ZIP entry")
    return digest.hexdigest()


def _wav_info(path: Path, row: dict, digest: str) -> dict:
    with wave.open(str(path), "rb") as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        duration = frames / rate
        if abs(duration - float(row["duration"])) > 1 / rate:
            raise ValueError(f"{row['RWCID']}: WAV duration differs from the complete catalog recording")
    return {"frames": frames, "sample_rate": rate, "duration_seconds": duration, "sha256": digest}


def _stage_member(archive: ZipFile, entry, row: dict, destination: Path, fetch) -> dict:
    if destination.exists():
        with destination.open("rb") as source:
            return _wav_info(destination, row, _check_copy(source, entry))

    fetch()
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".wav")
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as target, archive.open(entry) as source:
            digest = _check_copy(source, entry, target)
        info = _wav_info(temporary, row, digest)
        os.link(temporary, destination)
        return info
    finally:
        temporary.unlink()


def prepare(metadata: Path, output: Path) -> Path:
    candidates = select_candidates(metadata)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tracks = []
    for collection, size in ARCHIVE_SIZES.items():
        folder = f"RWC-{collection}"
        url = f"{RELEASE}/files/{folder}.zip?download=1"
        with tempfile.TemporaryFile(dir=output) as sparse:
            _copy_range(url, size, size - ZIP_TAIL_SIZE, size, sparse)
            with ZipFile(sparse) as archive:
                offsets = sorted(entry.header_offset for entry in archive.infolist()) + [archive.start_dir]
                member_stops = dict(zip(offsets, offsets[1:]))
                for row in candidates:
                    if row["CollID"] != collection:
                        continue
                    track_id = row["RWCID"]
                    # Exact member lookup avoids accepting an ambiguous or alternate mix.
                    entry = archive.getinfo(f"{folder}/{track_id}.wav")
                    destination = audio_dir / f"{track_id}.wav"
                    start, stop = entry.header_offset, member_stops[entry.header_offset]
                    info = _stage_member(
                        archive, entry, row, destination,
                        lambda: _copy_range(url, size, start, stop, sparse),
                    )
                    tracks.append(info | {
                        "track_id": track_id,
                        "audio": destination.relative_to(output).as_posix(),
                        "artist": row["Artist"],
                        "title": row["Title"],
                        "genre": row["genre"],
                        "rwc_genre": row["GenreMain"],
                        "rwc_subgenre": row["GenreSub"],
                        "condition": "original_mix",
                        "language": "en",
                        "archive_member": entry.filename,
                        "archive_url": url,
                    })
                    print(f"Staged {len(tracks)}/{len(candidates)}", flush=True)
    inventory = output / "inventory.json"
    payload = {
        "dataset": "RWC Music Database 2.0",
        "split": "test",
        "status": "candidate recordings pending lyric permission and independent lead-vocal reference review",
        "release": RELEASE,
        "license": "CC BY-NC 4.0",
        "license_url": LICENSE,
        "recording_permission": f"RWC Music Database noncommercial research release, {RELEASE}",
        "attribution": "Credit Goto et al. (2002, 2003) and the 2026 RWC Music Database re-release.",
        "metadata_url": METADATA,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "tracks": tracks,
    }
    fd, temporary = tempfile.mkstemp(dir=output, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, inventory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="local copy of the official RWC metadata.csv")
    parser.add_argument("output", type=Path, help="output directory for isolated test candidates")
    parser.add_argument("--plan", action="store_true", help="print selection counts without fetching audio")
    args = parser.parse_args()
    if args.plan:
        plan = select_candidates(args.metadata)
        print(json.dumps({
            "tracks": len(plan),
            "artists": len({row["Artist"] for row in plan}),
            "duration_seconds": sum(float(row["duration"]) for row in plan),
            "genres": dict(Counter(row["genre"] for row in plan)),
        }, indent=2))
    else:
        print(prepare(args.metadata, args.output))


if __name__ == "__main__":
    main()
