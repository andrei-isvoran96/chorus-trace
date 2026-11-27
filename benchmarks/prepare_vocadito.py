"""Stage the English clips in vocadito v3 as a training-only manifest."""

import argparse
import csv
import hashlib
import json
import wave
from pathlib import Path
from zipfile import ZipFile

from chorustrace.data import normalize_text, write_jsonl


RELEASE = "https://zenodo.org/records/5578807"
PERMISSION = f"Bittner et al., vocadito v3, CC BY 4.0, {RELEASE}"


def prepare(archive_path: Path, output: Path) -> Path:
    if hashlib.md5(archive_path.read_bytes()).hexdigest() != "dea40fd18f14d899643c4ba221b33a46":
        raise ValueError("archive does not match the published vocadito v3 checksum")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "train.jsonl"
    rows = []
    with ZipFile(archive_path) as archive:
        metadata = archive.read("vocadito_metadata.csv").decode("utf-8-sig").splitlines()
        for row in csv.DictReader(metadata):
            if row["language"] != "English":
                continue
            number = int(row["track_id"])
            name = f"vocadito_{number}"
            audio = output / f"{name}.wav"
            reference = output / f"{name}.json"
            text = archive.read(f"Annotations/Lyrics/{name}_lyrics.txt").decode("utf-8")
            reference.write_text(json.dumps({
                "text": normalize_text(text), "complete_lead_vocal": True,
                "transcribed_by": "vocadito annotators",
            }) + "\n", encoding="utf-8")
            audio.write_bytes(archive.read(f"Audio/{name}.wav"))
            with wave.open(str(audio), "rb") as wav:
                rows.append({
                    "track_id": f"vocadito:clip:{number}", "song_id": f"vocadito:work:{number}",
                    "artist_ids": [f"vocadito:{row['singer_id']}"], "lyric_id": f"vocadito:lyric:{number}",
                    "recording_family_id": f"vocadito:recording:{number}", "genre": "unclassified",
                    "split": "train", "audio": audio.name, "reference": reference.name,
                    "audio_source": RELEASE, "source_frames": wav.getnframes(),
                    "source_sample_rate": wav.getframerate(), "full_track": False,
                    "condition": "vocal", "language": "en", "recording_permission": PERMISSION,
                    "lyrics_permission": PERMISSION,
                })
    write_jsonl(manifest, rows)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(prepare(args.archive, args.output))
