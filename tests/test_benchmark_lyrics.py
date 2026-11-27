import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chorustrace.data import ALPHABET, Example, encode_text, write_jsonl
from scripts.benchmark_lyrics import (
    BenchmarkClip,
    audit_manifest,
    score,
    verify_checkpoint,
)


class BenchmarkTests(unittest.TestCase):
    def test_scoring_separates_instrumentals_and_counts_empty_voiced_drafts(self):
        clips = [
            BenchmarkClip(Example(Path("a"), "we go", "a", "test", "vocal", "one"), "a", "pop", "a", 2),
            BenchmarkClip(Example(Path("b"), "hi", "b", "test", "vocal", "one"), "b", "folk", "b", 1),
            BenchmarkClip(Example(Path("c"), "", "a", "test", "instrumental", "one"), "a", "pop", "a", 2),
            BenchmarkClip(Example(Path("d"), "", "b", "test", "instrumental", "one"), "b", "folk", "b", 1),
        ]
        predictions = {"a": ("we no", 0.9), "b": ("i", 0.6), "c": ("la", 0.9), "d": ("mu", 0.6)}
        calls = []

        def posteriors(path):
            calls.append(path.name)
            text, probability = predictions[path.name]
            labels = encode_text(text) or [0]
            result = np.full((len(labels), len(ALPHABET) + 1), (1 - probability) / len(ALPHABET))
            result[np.arange(len(labels)), labels] = probability
            return result

        report = score(SimpleNamespace(posteriors=posteriors), clips, samples=1, threshold=0.7)
        vocal = report["conditions"]["vocal"]
        instrumental = report["conditions"]["instrumental"]
        self.assertEqual(calls, ["a", "b", "c", "d"])
        self.assertEqual((vocal["reference_words"], vocal["word_edits"], vocal["wer"]), (3, 2, 2 / 3))
        self.assertEqual((vocal["reference_characters"], vocal["character_edits"], vocal["cer"]), (7, 3, 3 / 7))
        self.assertEqual(vocal["greedy"]["character_edits"], 2)
        self.assertEqual((vocal["voiced_rejects"], vocal["voiced_gate_suppressions"]), (1, 1))
        self.assertEqual(instrumental["instrumental_word_rate"], 0.5)
        self.assertIsNone(instrumental["wer"])
        self.assertEqual(report["genres"]["folk"]["vocal"]["samples"][0]["prediction"], "")
        self.assertEqual([result["prediction"] for result in report["false_alarms"]], ["la"])
        self.assertEqual([result["song_id"] for result in report["greedy_false_alarms"]], ["a", "b"])
        self.assertEqual(report["greedy_false_alarms"][1]["prediction"], "")

    def test_audit_rejects_cross_split_artist_and_duplicate_audio(self):
        rows = [make_row("first", "train", "singer"), make_row("second", "test", "SINGER")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "artist .* spans splits"):
                audit_manifest(path, {"first.wav": "a", "second.wav": "b"})
            rows[1]["artist"] = "another"
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate voiced audio spans splits"):
                audit_manifest(path, {"first.wav": "a", "second.wav": "a"})

    def test_audit_accepts_mix_only_and_rejects_disagreeing_paired_lyrics(self):
        rows = [make_row("first", "train", "a"), make_row("second", "test", "b", "original_mix")]
        digests = {"first.wav": "a", "second.wav": "b", "vocal.wav": "c"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(len(audit_manifest(path, digests)), 2)
            rows.append(dict(rows[1], condition="vocal", text="other lyric", audio="vocal.wav"))
            write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "unmatched lyric reference"):
                audit_manifest(path, digests)

    def test_checkpoint_provenance_is_bound_to_frozen_training_artists(self):
        train = BenchmarkClip(Example(Path("a"), "hi", "a", "train"), "a", "pop", "a", 1)
        test = BenchmarkClip(Example(Path("b"), "hello", "b", "test"), "b", "folk", "b", 1)
        checked = []
        recognizer = SimpleNamespace(
            training_manifest_sha256="frozen", training_song_ids=("a",), training_artist_ids=("a",),
            check_held_out=checked.extend,
        )
        verify_checkpoint(recognizer, [train, test], "frozen")
        self.assertEqual(checked, [test.example])
        with self.assertRaisesRegex(ValueError, "missing provenance"):
            verify_checkpoint(recognizer, [train, test], "changed")
        recognizer.training_artist_ids = ("b",)
        with self.assertRaisesRegex(ValueError, "training artist IDs disagree"):
            verify_checkpoint(recognizer, [train, test], "frozen")


def make_row(song, split, artist, condition="vocal"):
    return {
        "song_id": song, "clip_id": "one", "split": split, "artist": artist, "genre": "pop",
        "recording_id": song, "condition": condition, "audio": f"{song}.wav", "text": "our lyric",
        "start": 0, "end": 1, "duration_seconds": 1,
    }
