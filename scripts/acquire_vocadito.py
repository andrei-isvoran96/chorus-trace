"""Fetch the pinned vocadito archive and export its reviewed English subset."""

import argparse
import csv
import shutil
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from chorustrace.data import file_sha256, write_jsonl

RECORD = "https://zenodo.org/records/5557945"
ARCHIVE_URL = f"{RECORD}/files/vocadito.zip"
ARCHIVE_SHA256 = "5cb1ac4b2083ac9af55cae9cf1273bde7a25a2173e5ffee03887e086bc0f0db9"
RECORDING_PERMISSION = (
    "Volunteer performance, recording, and human annotations released by the dataset "
    "creators under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)."
)

# Reviewed entries only. The Rattlin' Bog performances share a composition ID.
SONGS = {
    "rattlin-bog": ((10, 34), "folk", "https://glostrad.com/green-grass-grew-all-around-the/"),
    "clementine": ((11,), "folk", "https://www.themorgan.org/music-manuscripts-and-printed-music/140365"),
    "happy-birthday": ((14,), "nursery-rhyme", "https://www.wipo.int/en/web/wipo-magazine/articles/in-the-courts-court-confirms-legal-status-of-happy-birthday-to-you-55581"),
    "house-of-the-rising-sun": ((15,), "folk", "https://folkways-media.si.edu/docs/folkways/artwork/FW02363.pdf"),
    "take-me-out-to-the-ball-game": ((19,), "tin-pan-alley", "https://www.loc.gov/item/2005561767/"),
    "we-wish-you-a-merry-christmas": ((20,), "carol", "https://imslp.org/wiki/We_Wish_You_a_Merry_Christmas_(Warrell,_Arthur_Sydney)"),
    "drunken-sailor": ((21,), "folk", "https://www.loc.gov/item/afc9999005.12828/"),
    "baa-baa-black-sheep": ((32,), "nursery-rhyme", "https://rpo.library.utoronto.ca/content/bah-bah-black-sheep"),
    "up-on-the-housetop": ((38,), "carol", "https://imslp.org/wiki/Up_on_the_Housetop_(Hanby,_Benjamin_Russel)"),
    "hello-ma-baby": ((39,), "tin-pan-alley", "https://www.loc.gov/item/99470294/"),
}


def archive_path(output: Path) -> Path:
    archive = output / "vocadito.zip"
    if not archive.exists():
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            downloaded = Path(temporary) / archive.name
            request = Request(ARCHIVE_URL, headers={"User-Agent": "ChorusTrace data preparation"})
            with urlopen(request, timeout=60) as response, downloaded.open("wb") as target:
                shutil.copyfileobj(response, target)
            if file_sha256(downloaded) != ARCHIVE_SHA256:
                raise ValueError("downloaded vocadito archive failed SHA-256 verification")
            downloaded.replace(archive)
    if file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError(f"{archive}: vocadito archive failed SHA-256 verification")
    return archive


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    archive = archive_path(output)
    (output / "audio").mkdir(exist_ok=True)
    sources = []
    rights = []
    with ZipFile(archive) as data:
        metadata = csv.DictReader(data.read("vocadito_metadata.csv").decode("utf-8").splitlines())
        tracks = {int(row["track_id"]): row for row in metadata if row["language"] == "English"}
        for song, (ids, genre, lyric_source) in SONGS.items():
            rights_id = f"vocadito-{song}"
            rights.append({
                "rights_id": rights_id,
                "source": RECORD,
                "recording_permission": RECORDING_PERMISSION,
                "lyrics_permission": f"Traditional or public-domain words ({lyric_source}).",
                "training_allowed": True,
                "checkpoint_release_allowed": True,
                "notes": "See docs/data-rights.md for attribution and composition evidence.",
            })
            for track_id in ids:
                track = tracks[track_id]
                name = f"vocadito_{track_id}"
                wav = data.read(f"Audio/{name}.wav")
                with wave.open(BytesIO(wav)) as recording:
                    seconds = recording.getnframes() / recording.getframerate()
                text = data.read(f"Annotations/Lyrics/{name}_lyrics.txt").decode("utf-8").strip()
                relative_audio = f"audio/{name}.wav"
                (output / relative_audio).write_bytes(wav)
                sources.append({
                    "song_id": song,
                    "recording_id": name,
                    "clip_id": name,
                    "artist": f"vocadito-{track['singer_id']}",
                    "genre": genre,
                    "language": "en",
                    "annotation_verified": True,
                    "start": 0,
                    "end": seconds,
                    "text": text,
                    "vocal": relative_audio,
                    "rights_id": rights_id,
                    "average_pitch_midi": int(track["average_pitch"]),
                })
    write_jsonl(output / "sources.jsonl", sources)
    write_jsonl(output / "rights.jsonl", rights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
