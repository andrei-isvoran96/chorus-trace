"""Prepare selected clips from locally extracted MUSDB18-HQ and partial lyrics."""

import tempfile
from pathlib import Path

from .audio import read_wav, write_wav
from .data import (
    check_song_split,
    lyric_field,
    read_jsonl,
    split_field,
    string_field,
    write_jsonl,
)
from .prepare import DEFAULT_LEVELS, _clip_batch, prepare_benchmark


def _seconds(timestamp: str) -> int:
    minutes, seconds = timestamp.split(":")
    return 60 * int(minutes) + int(seconds)


def _lyrics(path: Path) -> dict:
    result = {}
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            fields = line.strip().split(maxsplit=3)
            if len(fields) != 4 or fields[0].startswith("*"):
                continue
            start, end, voice_class, text = fields
            if voice_class in ("a", "b"):
                result[_seconds(start), _seconds(end), voice_class] = text
    return result


def prepare_musdb18(
    audio_dir: str | Path, lyrics_dir: str | Path, selection: str | Path, output_dir: str | Path,
    *, levels: tuple[float, ...] = DEFAULT_LEVELS, mix_only: bool = False,
) -> Path:
    """Look up exact annotated intervals and render the requested benchmark WAVs."""
    audio_dir, lyrics_dir, output_dir = Path(audio_dir).resolve(), Path(lyrics_dir), Path(output_dir)
    rows = []
    annotation_cache = {}
    splits = {}
    seen = set()
    for location, mark in read_jsonl(Path(selection)):
        song_id = string_field(mark, "song_id", location)
        split = split_field(mark, location, required=True)
        if split not in ("train", "test"):
            raise ValueError(f"{location}: MUSDB18 collections are train and test")
        check_song_split(splits, song_id, split, location)
        start, end = mark.get("start"), mark.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError(f"{location}: start and end must be increasing integer seconds")
        voice_class = mark.get("voice_class", "a")
        if voice_class != "a" and not (mix_only and voice_class == "b"):
            raise ValueError(f"{location}: class b is only supported for the mix-only chorus check")
        clip_id = string_field(mark, "clip_id", location) if "clip_id" in mark else f"{start}-{end}"
        if (song_id, clip_id) in seen:
            raise ValueError(f"{location}: duplicate clip {song_id!r} / {clip_id!r}")
        seen.add((song_id, clip_id))

        key = split, song_id
        if key not in annotation_cache:
            annotation_cache[key] = _lyrics(lyrics_dir / f"{split}_lyrics" / f"{song_id}.txt")
        text = annotation_cache[key].get((start, end, voice_class))
        if text is None:
            raise ValueError(f"{location}: no unstarred class {voice_class} lyric at {start}–{end} s")
        text = lyric_field({"text": text}, location, allow_empty=False)
        track = audio_dir / split / song_id
        rows.append({
            "song_id": song_id, "clip_id": clip_id, "split": split, "text": text,
            "start": start, "end": end, "mix": str(track / "mixture.wav"),
            "vocal": str(track / "vocals.wav"),
            "accompaniment": [str(track / name) for name in ("drums.wav", "bass.wav", "other.wav")],
        })
    if not rows:
        raise ValueError(f"{selection}: no MUSDB18 clips selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    if mix_only:
        return _prepare_mixes(rows, output_dir)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        sources = Path(temporary) / "sources.jsonl"
        write_jsonl(sources, rows)
        return prepare_benchmark(sources, output_dir, levels)


def _prepare_mixes(rows: list[dict], output_dir: Path) -> Path:
    with _clip_batch(output_dir) as (batch, mixed):
        previous = None
        for index, row in enumerate(rows):
            if row["mix"] != previous:
                mix, rate = read_wav(row["mix"])
                previous = row["mix"]
            start, end = row["start"] * rate, row["end"] * rate
            if end > len(mix):
                raise ValueError(f"{row['mix']}: excerpt ending at {row['end']} s exceeds WAV duration")
            audio = batch / f"{index:04d}.wav"
            write_wav(audio, mix[start:end], rate)
            mixed.append({
                key: row[key] for key in ("text", "song_id", "clip_id", "split")
            } | {"audio": audio.relative_to(output_dir).as_posix(), "condition": "original_mix"})
    return output_dir / "manifest.jsonl"
