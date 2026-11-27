"""Probe an existing recognizer using continuous synthetic mixes and real prefixes."""

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from chorustrace.audio import read_wav, write_wav
from chorustrace.ctc import greedy_decode
from chorustrace.data import decode_characters, read_jsonl, read_manifest
from chorustrace.evaluation import ErrorCounts
from chorustrace.recognizer import Recognizer
from chorustrace.synthetic import make_synthetic_sources


def synthetic_probe(directory: Path) -> dict:
    checkpoint = Path("benchmarks/checkpoints/synthetic_baseline.npz")
    model = Recognizer.load(checkpoint)
    manifest = make_synthetic_sources(directory, seed=0)
    songs = defaultdict(list)
    for _, row in read_jsonl(manifest):
        if row["split"] == "test":
            songs[row["song_id"]].append(row)

    total, tracks = ErrorCounts(), []
    for song_id, rows in songs.items():
        segments, lyrics = [], []
        for row in rows:
            mix, rate = read_wav(directory / row["mix"])
            backing, _ = read_wav(directory / row["accompaniment"])
            segments.extend((backing[:rate], mix))
            lyrics.append(row["text"])
        segments.append(backing[:rate])
        samples = np.concatenate(segments)
        audio = directory / f"{song_id}.wav"
        write_wav(audio, samples, rate)
        reference = " ".join(lyrics)
        prediction = model.transcribe(audio)
        counts = ErrorCounts()
        counts.add(reference, prediction)
        total.add(reference, prediction)
        tracks.append({"track_id": song_id, "duration_seconds": len(samples) / rate,
                       "reference": reference, "prediction": prediction, **counts.report()})
    return {"checkpoint_sha256": sha256(checkpoint), "aggregate": total.report(), "tracks": tracks}


def prefix_probe(checkpoint: Path) -> dict:
    model = Recognizer.load(checkpoint)
    marks = Path("benchmarks/musdb18_genre20.jsonl")
    genres = {row["song_id"]: row["genre"] for _, row in read_jsonl(marks) if row["split"] == "test"}
    clips = defaultdict(list)
    held_out = [row for row in read_manifest("runs/musdb18-genre20/manifest.jsonl")
                if row.split == "test" and row.condition == "original_mix"]
    model.check_held_out(held_out)
    for row in held_out:
        clips[row.song_id].append(row)

    total, tracks = ErrorCounts(), []
    for song_id, rows in sorted(clips.items()):
        audio = Path("data/musdb18_genre20/audio/test") / song_id / "mixture.wav"
        probabilities = model.posteriors(audio)
        prediction = decode_characters(greedy_decode(probabilities))
        centers = (np.arange(len(probabilities)) * model.features.hop_size
                   + model.features.window_size / 2) / model.features.sample_rate
        # Slice only for scoring known intervals. Inference sees the continuous input.
        counts = ErrorCounts()
        for row in rows:
            start, end = map(int, row.clip_id.split("-"))
            sample = decode_characters(greedy_decode(probabilities[(centers >= start) & (centers < end)]))
            counts.add(row.text, sample)
            total.add(row.text, sample)
        tracks.append({"track_id": song_id, "genre": genres[song_id],
                       "annotated_intervals": len(rows), "emitted_words": len(prediction.split()),
                       "prediction_preview": " ".join(prediction.split()[:12]), **counts.report()})
    return {"checkpoint_sha256": sha256(checkpoint), "aggregate": total.report(), "tracks": tracks}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        report = {
            "eligible_full_tracks": 0,
            "goal_wer": None,
            "synthetic_fixture": synthetic_probe(Path(temporary)),
            "real_prefix_five_song_checkpoint": prefix_probe(Path("runs/musdb18/baseline.npz")),
            "real_prefix_twenty_song_checkpoint": prefix_probe(Path("runs/musdb18-genre20/baseline.npz")),
        }
    print(json.dumps(report, indent=2))
