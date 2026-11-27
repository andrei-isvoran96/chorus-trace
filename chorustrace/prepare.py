"""Render paired transcription conditions from time-aligned PCM stems."""

import math
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import read_wav, write_wav
from .data import (
    check_song_split,
    lyric_field,
    read_jsonl,
    split_field,
    string_field,
    write_jsonl,
)

DEFAULT_LEVELS = (-12, -6, 0, 6)


@dataclass(frozen=True)
class SourceClip:
    song_id: str
    clip_id: str
    split: str | None
    text: str
    tracks: tuple[Path | None, ...]
    bounds: tuple[float, float] | None
    metadata: dict = field(default_factory=dict)


def _read_sources(path: Path) -> list[SourceClip]:
    clips = []
    seen = set()
    splits = {}
    for location, row in read_jsonl(path):
        song = string_field(row, "song_id", location)
        clip = string_field(row, "clip_id", location)
        split = split_field(row, location, required=True)
        check_song_split(splits, song, split, location)
        if (song, clip) in seen:
            raise ValueError(f"{location}: duplicate clip {song!r} / {clip!r}")
        seen.add((song, clip))

        accompaniment = row.get("accompaniment")
        if isinstance(accompaniment, str):
            accompaniment = [accompaniment]
        if not isinstance(accompaniment, list) or not accompaniment or any(
            not isinstance(track, str) or not track.strip() for track in accompaniment
        ):
            raise ValueError(f"{location}: accompaniment must be a WAV path or list of WAV paths")
        tracks = (string_field(row, "vocal", location), string_field(row, "mix", location), *accompaniment)

        bounds = None
        if "start" in row or "end" in row:
            start, end = row.get("start"), row.get("end")
            if not all(type(value) in (int, float) and math.isfinite(value) for value in (start, end)) \
                    or not 0 <= start < end:
                raise ValueError(f"{location}: start and end must be increasing times in seconds")
            bounds = float(start), float(end)
        clips.append(SourceClip(
            song, clip, split, lyric_field(row, location, allow_empty=False),
            tuple(path.parent / track for track in tracks), bounds,
        ))
    if not clips:
        raise ValueError(f"{path}: source manifest is empty")
    return clips


def _load_tracks(paths: tuple[Path | None, ...], label: str):
    sources = [read_wav(path) if path is not None else None for path in paths]
    available = [source for source in sources if source is not None]
    first, rate = available[0]
    if any(other_rate != rate or len(samples) != len(first) for samples, other_rate in available[1:]):
        raise ValueError(f"{label}: WAV tracks must have the same sample rate and duration")
    return [source[0] if source is not None else None for source in sources], rate, len(first)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples))))


@contextmanager
def _clip_batch(output_dir: Path):
    clips = output_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    batch = clips / Path(tempfile.mkdtemp(dir=clips, prefix="batch-")).name
    rows = []
    try:
        yield batch, rows
        write_jsonl(output_dir / "manifest.jsonl", rows)
    except BaseException:
        shutil.rmtree(batch, ignore_errors=True)
        raise


def _mix_levels(levels):
    levels = tuple(float(level) if level else 0.0 for level in levels)
    if not all(math.isfinite(level) for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("accompaniment levels must be finite and distinct")
    names = [f"mix_{level:+g}dB" for level in levels]
    if len(set(names)) != len(names):
        raise ValueError("accompaniment levels are too close to label distinctly")
    try:
        ratios = [10 ** (level / 20) for level in levels]
    except OverflowError as error:
        raise ValueError("accompaniment level is outside the supported numeric range") from error
    if any(ratio == 0 for ratio in ratios):
        raise ValueError("accompaniment level is outside the supported numeric range")
    return list(zip(names, ratios))


def _render_conditions(vocal, original, backing, levels, label, check_alignment):
    signals = {}
    if vocal is not None:
        vocal_rms = _rms(vocal)
        if not vocal_rms:
            raise ValueError(f"{label}: vocal must have nonzero RMS")
        signals["vocal"] = vocal
    if original is not None:
        if check_alignment and not _rms(original):
            raise ValueError(f"{label}: original mix must have nonzero RMS")
        signals["original_mix"] = original
    if backing is not None:
        backing_rms = _rms(backing)
        if check_alignment and vocal is not None and original is not None:
            reconstructed = vocal + backing
            scale = _rms(reconstructed) * _rms(original)
            if not scale or np.mean(reconstructed * original) / scale < 0.95:
                raise ValueError(f"{label}: original mix and summed stems do not agree at zero lag")
        if not backing_rms:
            if check_alignment:
                return signals
            raise ValueError(f"{label}: accompaniment must have nonzero RMS")
        if vocal is not None:
            for name, ratio in levels:
                gain = (vocal_rms / backing_rms) * ratio
                if not math.isfinite(gain) or gain > np.finfo(np.float64).max / max(1, np.max(np.abs(backing))):
                    raise ValueError(f"{label}: accompaniment level is outside the supported numeric range")
                signals[name] = vocal + gain * backing
        signals["instrumental"] = backing
    return signals


def render_clips(clips, batch: Path, output_dir: Path, levels=DEFAULT_LEVELS, *, check_alignment=False):
    """Render supplied clips into an owned batch directory and return the manifest rows."""
    levels = _mix_levels(levels)
    rows = []
    previous_paths = None
    for index, clip in enumerate(clips):
        label = f"{clip.song_id}/{clip.clip_id}"
        if clip.tracks != previous_paths:
            tracks, rate, length = _load_tracks(clip.tracks, label)
            previous_paths = clip.tracks
        start, end = (0, length)
        if clip.bounds is not None:
            start, end = (round(seconds * rate) for seconds in clip.bounds)
        if not 0 <= start < end <= length:
            raise ValueError(f"{label}: clip bounds exceed the WAV duration or yield no samples")
        vocal, original, *parts = (track[start:end] if track is not None else None for track in tracks)
        backing = sum(parts) if parts else None
        signals = _render_conditions(vocal, original, backing, levels, label, check_alignment)
        peak = max(float(np.max(np.abs(samples))) for samples in signals.values())
        scale = min(1.0, (32767 / 32768) / peak)

        folder = batch / f"{index:04d}"
        folder.mkdir()
        for condition, samples in signals.items():
            audio = folder / f"{condition}.wav"
            write_wav(audio, scale * samples, rate)
            row = {
                "audio": audio.relative_to(output_dir).as_posix(),
                "text": "" if condition == "instrumental" else clip.text,
                "song_id": clip.song_id,
                "clip_id": clip.clip_id,
                "split": clip.split,
                "condition": condition,
            }
            if clip.metadata:
                row.update(clip.metadata, duration_seconds=(end - start) / rate)
            rows.append(row)
    return rows


def prepare_benchmark(
    sources: str | Path,
    output_dir: str | Path,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> Path:
    """Write a JSONL manifest with isolated, mixed, and accompaniment-only WAV clips."""
    clips = _read_sources(Path(sources))
    output_dir = Path(output_dir)
    with _clip_batch(output_dir) as (batch, rows):
        rows.extend(render_clips(clips, batch, output_dir, levels))
    return output_dir / "manifest.jsonl"
