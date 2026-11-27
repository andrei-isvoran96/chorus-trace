"""Load and audit song data, and freeze complete original-mix test tracks.

Keep separate manifests for development and test when practical. The lock
contains identifiers and digests, while test WAVs and reference JSON remain
with the independent evaluator. Permissions, source provenance and reviewer
records refer to evidence checked by the corpus owner.
"""

import hashlib
import json
import os
import struct
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import SPLITS, normalize_text, read_jsonl, split_field, string_field


@dataclass(frozen=True)
class TimedReferenceWord:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class FullTrack:
    track_id: str
    song_id: str
    artist_ids: tuple[str, ...]
    lyric_id: str
    recording_family_id: str
    genre: str
    split: str
    audio: Path
    text: str
    duration_seconds: float
    words: tuple[TimedReferenceWord, ...] = ()
    instrumental_intervals: tuple[tuple[float, float], ...] | None = None
    full_track: bool = True
    condition: str = "original_mix"
    diagnostic_only: bool = False


@dataclass(frozen=True)
class _LoadedTrack:
    track: FullTrack
    signature: dict


_ID_FIELDS = ("track_id", "song_id", "lyric_id", "recording_family_id")
_RECORDING_PROVENANCE = ("audio_source", "recording_permission")
_PROVENANCE = (*_RECORDING_PROVENANCE, "lyrics_permission")
_FILE_DIGEST_FIELDS = ("audio_sha256", "pcm_sha256", "reference_sha256")
_GROUP_FIELDS = (*_ID_FIELDS, "lyrics_sha256", "audio_sha256", "pcm_sha256")
_MIN_TEST_TRACKS = 20


def _identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _ids(row: dict, location: str) -> dict:
    artists = row.get("artist_ids")
    if not isinstance(artists, list) or not artists or any(
        not isinstance(artist, str) or not _identity(artist) for artist in artists
    ):
        raise ValueError(f"{location}: artist_ids must contain canonical artist IDs")
    return {name: _identity(string_field(row, name, location)) for name in _ID_FIELDS} | {
        "artist_ids": sorted(set(map(_identity, artists)))
    }


def _sha256(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def _audio_signature(path: Path, row: dict, location: str) -> tuple[float, str, str]:
    with path.open("rb") as stream:
        try:
            with wave.open(stream, "rb") as wav:
                frames, sample_rate = wav.getnframes(), wav.getframerate()
                channels, width = wav.getnchannels(), wav.getsampwidth()
                if not frames or not sample_rate or width not in (1, 2, 3, 4):
                    raise ValueError(f"{path}: expected nonempty 8-, 16-, 24- or 32-bit integer PCM")
                for field, actual in (("source_frames", frames), ("source_sample_rate", sample_rate)):
                    expected = row.get(field)
                    if type(expected) is not int or expected != actual:
                        raise ValueError(f"{location}: {field} must match the supplied source WAV ({actual})")

                # Hash PCM and its interpretation separately from the RIFF container.
                pcm = hashlib.sha256(struct.pack("<IHH", sample_rate, channels, width))
                frame_size = channels * width
                batch_frames = (1 << 20) // frame_size
                for offset in range(0, frames, batch_frames):
                    count = min(batch_frames, frames - offset)
                    block = wav.readframes(count)
                    if len(block) != count * frame_size:
                        raise ValueError(f"{path}: incomplete PCM WAV")
                    pcm.update(block)
        except (EOFError, wave.Error) as error:
            raise ValueError(f"{path}: expected an integer PCM WAV ({error})") from error
        stream.seek(0)
        return frames / sample_rate, _sha256(stream), pcm.hexdigest()


def _interval(start, end, minimum: float, duration: float, location: str) -> tuple[float, float]:
    for value in (start, end):
        if type(value) not in (int, float) or not 0 <= value <= duration:
            raise ValueError(f"{location}: timestamps must be numbers inside the full track")
    if start < minimum or start >= end:
        raise ValueError(f"{location}: expect positive, ordered, non-overlapping intervals")
    return float(start), float(end)


def _reference_words(
    reference: dict, text: str, duration: float, location: str
) -> tuple[TimedReferenceWord, ...]:
    if "words" not in reference:
        return ()
    entries = reference["words"]
    if not isinstance(entries, list):
        raise ValueError(f"{location}: words must be a list of timed words")
    words = []
    previous_end = 0.0
    for index, entry in enumerate(entries, start=1):
        label = f"{location}: word {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label}: expected an object with word, start and end")
        word = normalize_text(string_field(entry, "word", label))
        if len(word.split()) != 1:
            raise ValueError(f"{label}: expected a single word")
        start, end = _interval(entry.get("start"), entry.get("end"), previous_end, duration, label)
        words.append(TimedReferenceWord(word, start, end))
        previous_end = end
    if " ".join(word.word for word in words) != text:
        raise ValueError(f"{location}: timed words must match the complete reference text in order")
    return tuple(words)


def _instrumental_intervals(
    reference: dict, words: tuple[TimedReferenceWord, ...], duration: float, location: str
) -> tuple[tuple[float, float], ...] | None:
    if "instrumental_intervals" not in reference:
        return None
    entries = reference["instrumental_intervals"]
    if not isinstance(entries, list):
        raise ValueError(f"{location}: instrumental_intervals must be a list of [start, end] pairs")
    intervals = []
    previous_end, word_index = 0.0, 0
    for index, entry in enumerate(entries, start=1):
        label = f"{location}: instrumental interval {index}"
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"{label}: expected a [start, end] pair")
        start, end = _interval(*entry, previous_end, duration, label)
        while word_index < len(words) and words[word_index].end <= start:
            word_index += 1
        if word_index < len(words) and words[word_index].start < end:
            raise ValueError(f"{label}: overlaps a lead-vocal word")
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def _load_track(manifest: Path, location: str, row: dict, split: str) -> _LoadedTrack:
    condition, full_track = row.get("condition"), row.get("full_track")
    diagnostic_only = row.get("diagnostic_only", False)
    if type(diagnostic_only) is not bool or (diagnostic_only and split != "test"):
        raise ValueError(f"{location}: diagnostic_only must be a boolean and may only be true for test rows")
    full_original = full_track is True and condition == "original_mix"
    training_clip = split == "train" and full_track is False and condition in ("original_mix", "vocal")
    validation_clip = split == "validation" and full_track is False and condition == "original_mix"
    instrumental_training = split == "train" and full_track is True and condition == "instrumental"
    no_lyrics = diagnostic_only or instrumental_training
    if (not no_lyrics and row.get("language") != "en") or not (
        full_original or training_clip or validation_clip or instrumental_training
    ):
        raise ValueError(
            f"{location}: expected a complete English-language original mix or an eligible development clip"
        )
    ids = _ids(row, location)
    provenance_fields = _RECORDING_PROVENANCE if no_lyrics else _PROVENANCE
    provenance = {field: string_field(row, field, location).strip() for field in provenance_fields}
    genre = _identity(string_field(row, "genre", location))
    if genre in ("hip hop", "hiphop", "rap"):
        genre = "hip-hop"
    audio_path = manifest.parent / string_field(row, "audio", location)
    reference_path = manifest.parent / string_field(row, "reference", location)
    reference_bytes = reference_path.read_bytes()
    try:
        reference = json.loads(reference_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{reference_path}: expected UTF-8 JSON ({error})") from error
    if not isinstance(reference, dict):
        raise ValueError(f"{reference_path}: expected a reference object")
    if split == "test" or instrumental_training:
        checked_by = _identity(string_field(reference, "checked_by", str(reference_path)))
        string_field(reference, "review_evidence", str(reference_path))
    if no_lyrics:
        if reference.get("text", "") != "" or reference.get("words", []) != []:
            scope = "diagnostic-only" if diagnostic_only else "instrumental training"
            raise ValueError(f"{reference_path}: {scope} references cannot contain lyric text or timed words")
        text = ""
    else:
        if reference.get("complete_lead_vocal") is not True:
            raise ValueError(f"{reference_path}: expected a complete lead-vocal reference")
        transcribed_by = _identity(string_field(reference, "transcribed_by", str(reference_path)))
        if split == "test" and checked_by == transcribed_by:
            raise ValueError(f"{reference_path}: a separate reviewer must check the reference")
        text = normalize_text(string_field(reference, "text", str(reference_path)))
        if not text:
            raise ValueError(f"{reference_path}: lead-vocal reference must contain words")
    duration, audio_hash, pcm_hash = _audio_signature(audio_path, row, location)
    words = _reference_words(reference, text, duration, str(reference_path))
    instrumental = _instrumental_intervals(reference, words, duration, str(reference_path))
    if diagnostic_only and not instrumental:
        raise ValueError(f"{reference_path}: diagnostic-only tracks require checked instrumental passages")
    if instrumental_training and not any(end - start >= 0.3 for start, end in (instrumental or ())):
        raise ValueError(f"{reference_path}: instrumental training requires a checked passage of at least 0.3 s")
    track = FullTrack(
        ids["track_id"], ids["song_id"], tuple(ids["artist_ids"]), ids["lyric_id"],
        ids["recording_family_id"], genre, split, audio_path, text, duration, words, instrumental,
        full_track, condition, diagnostic_only,
    )
    signature = ids | provenance | {
        "genre": genre,
        "audio_sha256": audio_hash,
        "pcm_sha256": pcm_hash,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
    }
    if diagnostic_only:
        signature["diagnostic_only"] = True
    elif not instrumental_training:
        signature["lyrics_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _LoadedTrack(track, signature)


def _load_selected(path: str | Path, splits: tuple[str, ...]) -> list[_LoadedTrack]:
    path = Path(path)
    return [
        _load_track(path, location, row, split)
        for location, row in read_jsonl(path)
        if (split := split_field(row, location, required=True)) in splits
    ]


def _check_splits(entries) -> None:
    seen = {}
    for split, signature in entries:
        groups = [(field, signature[field]) for field in _GROUP_FIELDS if field in signature]
        groups.extend(("artist_ids", artist) for artist in signature["artist_ids"])
        for field, identifier in groups:
            prior_split = seen.get((field, identifier))
            if prior_split is not None and (
                prior_split != split or field in ("track_id", "audio_sha256", "pcm_sha256")
                or (split == "test" and field != "artist_ids")
            ):
                raise ValueError(f"{split} track {signature['track_id']!r}: {field} overlaps the {prior_split} set")
            seen[field, identifier] = split


def _check_loaded(entries: list[_LoadedTrack]) -> None:
    _check_splits((entry.track.split, entry.signature) for entry in entries)


def _test_signatures(entries: list[_LoadedTrack]) -> list[dict]:
    lexical = [entry for entry in entries if not entry.track.diagnostic_only]
    if len(lexical) < _MIN_TEST_TRACKS:
        raise ValueError(f"frozen test requires at least {_MIN_TEST_TRACKS} complete lexical tracks, found {len(lexical)}")
    genres = {entry.track.genre for entry in lexical}
    required = {"pop", "rock", "hip-hop"}
    if not required <= genres or not genres - required:
        raise ValueError("frozen test requires pop, rock, hip-hop and at least one other genre")
    if not any(entry.track.instrumental_intervals for entry in entries):
        raise ValueError("frozen test requires annotated instrumental passages")
    return sorted((entry.signature for entry in entries), key=lambda item: item["track_id"])


def _read_lock(lock_path: str | Path) -> list[dict]:
    location = str(lock_path)
    with Path(lock_path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        not isinstance(payload, dict) or type(payload.get("version")) is not int
        or payload["version"] != 1 or not isinstance(payload.get("tracks"), list)
    ):
        raise ValueError(f"{location}: expected a version 1 full-track test lock")
    entries = payload["tracks"]
    for item in entries:
        if not isinstance(item, dict) or _ids(item, location) != {key: item.get(key) for key in (*_ID_FIELDS, "artist_ids")}:
            raise ValueError(f"{location}: invalid identifiers in lock")
        if "diagnostic_only" in item and item["diagnostic_only"] is not True:
            raise ValueError(f"{location}: invalid diagnostic-only marker in lock")
        diagnostic_only = item.get("diagnostic_only", False)
        provenance_fields = _RECORDING_PROVENANCE if diagnostic_only else _PROVENANCE
        for field in ("genre", *provenance_fields):
            string_field(item, field, location)
        digest_fields = _FILE_DIGEST_FIELDS if diagnostic_only else (*_FILE_DIGEST_FIELDS, "lyrics_sha256")
        for field in digest_fields:
            digest = item.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{location}: invalid {field}")
    if sum(not item.get("diagnostic_only", False) for item in entries) < _MIN_TEST_TRACKS:
        raise ValueError(f"{location}: incomplete frozen test")
    return entries


def read_fulltrack_manifest(path: str | Path) -> list[FullTrack]:
    """Read all splits and reject repeated tracks and cross-split identities."""
    entries = _load_selected(path, SPLITS)
    if not entries:
        raise ValueError(f"{path}: empty full-track manifest")
    _check_loaded(entries)
    return [entry.track for entry in entries]


def read_fulltrack_development(manifest_path: str | Path, lock_path: str | Path) -> list[FullTrack]:
    """Read train and validation assets and check their identities against the frozen test."""
    test = _read_lock(lock_path)
    entries = _load_selected(manifest_path, ("train", "validation"))
    if {entry.track.split for entry in entries} != {"train", "validation"}:
        raise ValueError(f"{manifest_path}: supply training assets and original-mix validation assets")
    groups = [("test", signature) for signature in test]
    groups.extend((entry.track.split, entry.signature) for entry in entries)
    _check_splits(groups)
    return [entry.track for entry in entries]


def freeze_fulltrack_test(manifest_path: str | Path, lock_path: str | Path) -> list[FullTrack]:
    """Create a new test lock without overwriting an existing lock."""
    entries = _load_selected(manifest_path, SPLITS)
    _check_loaded(entries)
    test = [entry for entry in entries if entry.track.split == "test"]
    payload = {
        "version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "tracks": _test_signatures(test),
    }
    lock_path = Path(lock_path)
    fd, temporary = tempfile.mkstemp(dir=lock_path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, lock_path)
    finally:
        Path(temporary).unlink()
    return sorted((entry.track for entry in test), key=lambda track: track.track_id)


def verify_fulltrack_test(manifest_path: str | Path, lock_path: str | Path) -> list[FullTrack]:
    """Verify test assets against the lock without opening development audio or references."""
    frozen = _read_lock(lock_path)
    test = _load_selected(manifest_path, ("test",))
    _check_loaded(test)
    if _test_signatures(test) != frozen:
        raise ValueError("test manifest or assets differ from the frozen test lock")
    return sorted((entry.track for entry in test), key=lambda track: track.track_id)
