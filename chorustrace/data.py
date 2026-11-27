"""Lyric normalization and song-grouped JSONL manifests."""

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ALPHABET = " abcdefghijklmnopqrstuvwxyz'"
SPLITS = ("train", "validation", "test")
CHARACTER_IDS = {character: index for index, character in enumerate(ALPHABET, start=1)}
_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)*")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold()).replace("’", "'").replace("‘", "'")
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(_WORDS.findall(text))


def encode_text(text: str) -> list[int]:
    return [CHARACTER_IDS[character] for character in normalize_text(text)]


def decode_characters(ids: list[int]) -> str:
    return normalize_text("".join(ALPHABET[index - 1] for index in ids))


@dataclass(frozen=True)
class Example:
    audio: Path
    text: str
    song_id: str
    split: str | None = None
    condition: str = "audio"
    clip_id: str | None = None
    artist: str | None = None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: {error.msg}") from error
            if not isinstance(item, dict):
                raise TypeError(f"{location}: expected a JSON object")
            yield location, item


def write_jsonl(path: Path, rows) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def string_field(row: dict, name: str, location: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {name} must be a nonempty string")
    return value


def split_field(row: dict, location: str, *, required: bool = False) -> str | None:
    value = row.get("split")
    if (required or value is not None) and value not in SPLITS:
        raise ValueError(f"{location}: split must be one of {', '.join(SPLITS)}")
    return value


def lyric_field(row: dict, location: str, *, allow_empty: bool = True) -> str:
    value = row.get("text")
    if not isinstance(value, str):
        raise TypeError(f"{location}: text must be a string")
    text = normalize_text(value)
    if not text and (not allow_empty or value.strip()):
        raise ValueError(f"{location}: text has no supported English characters")
    return text


def check_song_split(song_splits: dict, song_id: str, split: str | None, location: str) -> None:
    if song_splits.setdefault(song_id, split) != split:
        raise ValueError(f"{location}: song {song_id!r} occurs in multiple splits")


def read_manifest(path: str | Path) -> list[Example]:
    path = Path(path)
    examples = []
    song_splits = {}
    for location, row in read_jsonl(path):
        song_id = string_field(row, "song_id", location)
        split = split_field(row, location)
        check_song_split(song_splits, song_id, split, location)
        examples.append(Example(
            audio=path.parent / string_field(row, "audio", location),
            text=lyric_field(row, location),
            song_id=song_id,
            split=split,
            condition=string_field(row, "condition", location) if "condition" in row else "audio",
            clip_id=string_field(row, "clip_id", location) if "clip_id" in row else None,
            artist=string_field(row, "artist", location) if "artist" in row else None,
        ))
    if not examples:
        raise ValueError(f"{path}: manifest is empty")
    if None in song_splits.values() and any(split is not None for split in song_splits.values()):
        raise ValueError(f"{path}: provide split on every row or omit it throughout")
    return examples


def select_split(examples: list[Example], split: str) -> list[Example]:
    if split == "all" or all(example.split is None for example in examples):
        selected = examples
    else:
        selected = [example for example in examples if example.split == split]
    if not selected:
        raise ValueError(f"manifest has no examples in split {split!r}")
    return selected
