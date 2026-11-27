"""Reproduce a generated-only full-track development and checkpoint-reload run."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from chorustrace.audio import write_wav
from chorustrace.fulltrack_data import FullTrack, TimedReferenceWord
from chorustrace.fulltrack_evaluation import evaluate_full_tracks
from chorustrace.fulltrack_training import train_full_tracks
from chorustrace.recognizer import Recognizer, TrainingConfig
from chorustrace.synthetic import _MELODIES, _SAMPLE_RATE, _backing, _rms, _sing


_LYRICS = {
    "train": (
        ("see the light today", "we sing with you", "say a note tonight"),
        ("hello we sing today", "see you with me", "hear a light tonight"),
        ("sing with you tonight", "the moon is high", "see a song today"),
        ("hear a note today", "hello moon hello sky", "we see the light"),
        ("say a note today", "the sky is high", "hear me sing tonight"),
        ("see the moon tonight", "hear you with me", "we sing a note"),
        ("we see a light", "hello song hello note", "hear the sky today"),
        ("a song with you", "the light is high", "see me sing today"),
    ),
    "validation": (
        ("hear the moon today", "we see you tonight", "hello light hello moon"),
        ("see a note today", "we hear you sing", "the song is high"),
    ),
}


def _make_track(output, split, number, phrases, rng):
    track_id = f"generated-{split}-{number:02d}"
    key = int(rng.integers(51, 66))
    tempo = rng.uniform(90, 132)
    timbre = rng.uniform(0.87, 1.15)
    melody = _MELODIES[number % len(_MELODIES)]
    position = 0
    parts = []
    references = []
    instrumental = []

    def append_instrumental(seconds):
        nonlocal position
        start = position / _SAMPLE_RATE
        clip = np.zeros(round(seconds * _SAMPLE_RATE))
        parts.append(clip)
        position += len(clip)
        instrumental.append((start, position / _SAMPLE_RATE))

    append_instrumental(4.0)
    ordered_phrases = (phrases[0], phrases[1], phrases[0], phrases[2], phrases[1])
    for index, (phrase, pause) in enumerate(zip(ordered_phrases, (1.35, 4.75, 1.70, 6.85, 4.0))):
        for word in phrase.split():
            rotation = len(references) % len(melody)
            pattern = melody[rotation:] + melody[:rotation]
            vocal = _sing(word, key + index % 3, pattern, timbre, rng)
            nonzero = np.flatnonzero(vocal)
            references.append(TimedReferenceWord(
                word, (position + nonzero[0]) / _SAMPLE_RATE, (position + nonzero[-1] + 1) / _SAMPLE_RATE,
            ))
            parts.append(vocal)
            position += len(vocal)
        append_instrumental(pause)

    lead = np.concatenate(parts)
    backing = _backing(len(lead), key, tempo, rng)
    backing *= _rms(lead) * rng.uniform(0.65, 1.20) / _rms(backing)
    mix = np.column_stack((lead + backing, 0.94 * lead + 1.04 * backing))
    mix *= 0.90 / max(0.90, np.abs(mix).max())
    audio = output / f"{track_id}.wav"
    write_wav(audio, mix, _SAMPLE_RATE)
    return FullTrack(
        track_id, track_id, (track_id,), track_id, track_id, "synthetic", split, audio,
        " ".join(item.word for item in references), position / _SAMPLE_RATE,
        tuple(references), tuple(instrumental),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--reload", action="store_true", help="score generated validation from the saved checkpoint")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="chorustrace-generated-") as directory:
        rng = np.random.default_rng(2026)
        tracks = [
            _make_track(Path(directory), split, number, phrases, rng)
            for split, songs in _LYRICS.items()
            for number, phrases in enumerate(songs, 1)
        ]
        if args.reload:
            recognizer = Recognizer.load(args.checkpoint)
            validation = [track for track in tracks if track.split == "validation"]
            result = evaluate_full_tracks(recognizer, validation, compare_unsegmented=True)
        else:
            result = train_full_tracks(
                tracks, args.checkpoint, config=TrainingConfig(epochs=args.epochs),
                scale_groups=(2, 4, 8), validate_every=args.validate_every, progress=print,
            )
    result["scope"] = "generated full-track development validation"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
