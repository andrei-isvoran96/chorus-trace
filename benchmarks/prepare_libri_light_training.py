"""Stage Libri-Light's official one-hour orthographic subset as auxiliary training speech."""

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
import wave
from pathlib import Path

from benchmarks.prepare_clip1_training import BLOCK_SIZE, _decode_flac, _file_hexdigest
from chorustrace.audio import FeatureConfig
from chorustrace.ctc import minimum_frames
from chorustrace.data import encode_text, normalize_text, write_jsonl
from chorustrace.fulltrack_data import read_fulltrack_manifest, verify_fulltrack_test


RELEASE = "https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md"
ARCHIVE = "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
ARCHIVE_SHA256 = "5d1efdc777b548194d7e09ba89126e2188026df9fd57aa57eb14408d2b2342af"
PERMISSION = f"Libri-Light limited-supervision LibriSpeech subset, CC BY 4.0, https://www.openslr.org/12/ , {RELEASE}"
_PREFIX = "librispeech_finetuning/1h/"
_MEMBER = re.compile(r"([0-5])/(clean|other)/([0-9]+)/([0-9]+)/([0-9-]+)(\.flac|\.trans\.txt)")
_HOPS = (FeatureConfig(hop_size=320), FeatureConfig(window_size=512, hop_size=480))


def _read_sources(archive: Path, source_dir: Path) -> tuple[dict, dict]:
    audio, transcripts = {}, {}
    transcript_members = set()
    with gzip.open(archive, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as contents:
            for entry in contents:
                if not entry.name.startswith(_PREFIX) or entry.isdir():
                    continue
                match = _MEMBER.fullmatch(entry.name.removeprefix(_PREFIX))
                if match is None or not entry.isfile() or not 0 < entry.size <= 16 * BLOCK_SIZE:
                    raise ValueError(f"{entry.name}: invalid one-hour subset member")
                fold, condition, speaker, chapter, stem, suffix = match.groups()
                with contents.extractfile(entry) as member:
                    original = member.read()
                if len(original) != entry.size:
                    raise ValueError(f"{entry.name}: truncated tar member")
                group = f"{speaker}-{chapter}"
                if suffix == ".flac":
                    if not re.fullmatch(rf"{group}-[0-9]+", stem) or stem in audio:
                        raise ValueError(f"{entry.name}: invalid or duplicate utterance ID")
                    (source_dir / f"{stem}.flac").write_bytes(original)
                    audio[stem] = entry.name, fold, condition, speaker, chapter
                    continue
                if stem != group or group in transcript_members:
                    raise ValueError(f"{entry.name}: invalid or duplicate transcript member")
                transcript_members.add(group)
                for line in original.decode("utf-8").splitlines():
                    utterance, separator, text = line.partition(" ")
                    if not separator or not re.fullmatch(rf"{group}-[0-9]+", utterance) or utterance in transcripts:
                        raise ValueError(f"{entry.name}: invalid or duplicate transcript utterance ID")
                    if not normalize_text(text):
                        raise ValueError(f"{entry.name}: empty normalized transcript")
                    transcripts[utterance] = text
        # The gzip checksum is checked once the compressed stream reaches EOF.
        while stream.read(BLOCK_SIZE):
            pass
    if not audio or audio.keys() != transcripts.keys():
        raise ValueError(f"{archive}: audio and orthographic transcript IDs do not match")
    return audio, transcripts


def _six_word_slices(text: str):
    words = text.split()
    return (tuple(words[start:start + 6]) for start in range(len(words) - 5))


def _row(utterance: str, source: tuple, paths: tuple[str, str]) -> dict:
    member, fold, quality, speaker, chapter = source
    work = f"librispeech:chapter:{chapter}"
    return {
        "track_id": f"librispeech:{utterance}", "song_id": work, "lyric_id": work,
        "recording_family_id": f"librispeech:recording:{speaker}-{chapter}",
        "artist_ids": [f"librispeech:speaker:{speaker}"],
        "genre": "speech", "auxiliary_type": "read_speech", "source_subset": f"1h/{fold}/{quality}",
        "split": "train", "condition": "vocal", "language": "en", "full_track": False,
        "audio": paths[0], "reference": paths[1], "audio_source": f"{ARCHIVE} :: {member}",
        "recording_permission": PERMISSION, "lyrics_permission": PERMISSION,
    }


def prepare(
    archive: Path, validation: Path, test_manifest: Path, test_lock: Path, output: Path, *,
    ffmpeg: str = "ffmpeg", max_seconds: float = 40,
) -> Path:
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("maximum clip length must be finite and positive")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    archive_sha = _file_hexdigest(archive, hashlib.sha256())
    if archive_sha != ARCHIVE_SHA256:
        raise ValueError(f"{archive}: SHA-256 differs from the official Libri-Light archive")
    validate = read_fulltrack_manifest(validation)
    if any(track.split != "validation" for track in validate):
        raise ValueError(f"{validation}: supply a separate validation-only manifest")
    protected = validate + verify_fulltrack_test(test_manifest, test_lock)
    identity_fields = ("track_id", "song_id", "lyric_id", "recording_family_id")
    protected_ids = {field: {getattr(track, field) for track in protected} for field in identity_fields}
    protected_artists = {artist for track in protected for artist in track.artist_ids}
    protected_texts = {track.text for track in protected if track.text}
    protected_slices = {phrase for text in protected_texts for phrase in _six_word_slices(text)}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        root = Path(temporary)
        sources, staged = root / "source-flac", root / "dataset"
        sources.mkdir()
        (staged / "audio").mkdir(parents=True)
        (staged / "references").mkdir()
        audio, transcripts = _read_sources(archive, sources)
        rows = []
        overlap_count = 0
        fits = {str(config.hop_size): 0 for config in _HOPS}
        windows = {str(config.hop_size): config.window_size for config in _HOPS}
        for utterance in sorted(audio):
            audio_name, reference_name = f"audio/{utterance}.wav", f"references/{utterance}.json"
            row = _row(utterance, audio[utterance], (audio_name, reference_name))
            text = normalize_text(transcripts[utterance])
            if (
                any(row[field] in protected_ids[field] for field in identity_fields)
                or protected_artists.intersection(row["artist_ids"])
                or text in protected_texts
                or any(phrase in protected_slices for phrase in _six_word_slices(text))
            ):
                overlap_count += 1
                continue

            source = (sources / f"{utterance}.flac").read_bytes()
            wav_path = staged / audio_name
            pcm_md5 = _decode_flac(source, wav_path, ffmpeg, max_seconds)
            with wave.open(str(wav_path), "rb") as wav:
                frames, sample_rate = wav.getnframes(), wav.getframerate()
                if sample_rate != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError(f"{utterance}: expected mono 16 kHz PCM16 LibriSpeech audio")
            required = minimum_frames(encode_text(text))
            available = {}
            for config in _HOPS:
                hop = config.hop_size
                count = 1 + (max(0, frames - config.window_size) + hop - 1) // hop
                available[str(hop)] = count
                fits[str(hop)] += count >= required
            if available["320"] < required:
                wav_path.unlink()
                continue
            (staged / reference_name).write_text(json.dumps({
                "text": transcripts[utterance], "complete_lead_vocal": True,
                "transcribed_by": "LibriSpeech published orthographic transcript",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            rows.append(row | {
                "source_frames": frames, "source_sample_rate": sample_rate,
                "source_flac_sha256": hashlib.sha256(source).hexdigest(), "source_pcm_md5": pcm_md5,
                "audio_sha256": _file_hexdigest(wav_path, hashlib.sha256()),
                "ctc_required_frames": required, "ctc_available_frames": available,
            })
        if not rows:
            raise ValueError("no disjoint Libri-Light examples fit the hop-320 CTC feature grid")
        manifest = staged / "train.jsonl"
        write_jsonl(manifest, rows)
        tracks = read_fulltrack_manifest(manifest)
        report = {
            "official_subset": "1h", "archive_sha256": archive_sha, "verified_gzip_crc": True,
            "test_lock_sha256": _file_hexdigest(test_lock, hashlib.sha256()),
            "validation_manifest_sha256": _file_hexdigest(validation, hashlib.sha256()),
            "manifest_sha256": _file_hexdigest(manifest, hashlib.sha256()),
            "source_utterances": len(audio), "held_out_overlap_exclusions": overlap_count,
            "staged_utterances": len(rows), "ctc_fit_counts": fits, "feature_window_samples": windows,
            "staged_duration_seconds": sum(track.duration_seconds for track in tracks),
            "staged_maximum_seconds": max(track.duration_seconds for track in tracks),
            "staged_speakers": len({track.artist_ids[0] for track in tracks}),
            "staged_chapters": len({track.song_id for track in tracks}),
        }
        (staged / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staged, output)
    return output / "train.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "validation", "test_manifest", "test_lock", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--max-clip-seconds", type=float, default=40)
    args = parser.parse_args()
    print(prepare(args.archive, args.validation, args.test_manifest, args.test_lock, args.output,
                  ffmpeg=args.ffmpeg, max_seconds=args.max_clip_seconds))
