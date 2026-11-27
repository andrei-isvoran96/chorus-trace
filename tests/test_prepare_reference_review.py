import hashlib
import json
import shlex
import tempfile
import unittest
import wave
from pathlib import Path

from benchmarks.prepare_reference_review import prepare


class ReferenceReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "candidate track.wav"
        with wave.open(str(self.audio), "wb") as wav:
            wav.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes([137]) * 4000)
        self.digest = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        self.output = self.root / "review"
        self.inventory = self.root / "candidate.json"

    def write_inventory(self, **overrides):
        row = {"audio": self.audio.name, "frames": 4000, "sample_rate": 8000,
               "duration_seconds": 0.5, "sha256": self.digest} | overrides
        self.inventory.write_text(json.dumps({"tracks": [row]}), encoding="utf-8")

    def test_prepares_separate_blank_passes_without_overwriting_reviews(self):
        self.write_inventory()
        self.assertEqual(prepare(self.inventory, self.output), self.output)
        a, b = (self.output / label for label in ("pass-a", "pass-b"))
        [record] = json.loads((a / "tracks.json").read_text(encoding="utf-8"))
        self.assertEqual((a / "tracks.json").read_bytes(), (b / "tracks.json").read_bytes())
        self.assertEqual(record["audio"], str(self.audio.resolve()))
        self.assertEqual(record["duration_seconds"], 0.5)
        self.assertEqual(record["audio_sha256"], self.digest)
        self.assertEqual(shlex.split(record["playback"])[-1], str(self.audio.resolve()))
        first, second = a / record["reference"], b / record["reference"]
        blank = json.loads(first.read_text(encoding="utf-8"))
        self.assertFalse(blank.pop("complete_lead_vocal"))
        self.assertTrue(all(value is None for value in blank.values()))
        first.write_text("reviewer draft", encoding="utf-8")
        self.assertNotEqual(first.read_bytes(), second.read_bytes())
        with self.assertRaises(FileExistsError):
            prepare(self.inventory, self.output)
        self.assertEqual(first.read_text(encoding="utf-8"), "reviewer draft")

    def test_rejects_changed_wav_hash_or_duration_before_writing_packets(self):
        for field, value, message in (
            ("sha256", "a" * 64, "WAV SHA-256"),
            ("duration_seconds", 0.4, "WAV duration"),
            ("frames", 3000, "source_frames"),
        ):
            with self.subTest(field=field):
                self.write_inventory(**{field: value})
                with self.assertRaisesRegex(ValueError, message):
                    prepare(self.inventory, self.output)
                self.assertFalse(self.output.exists())

    def test_accepts_jsonl_candidates_with_source_frame_counts(self):
        source = self.root / "candidates.jsonl"
        row = {"audio": self.audio.name, "source_frames": 4000, "source_sample_rate": 8000,
               "track_id": "artist:work", "audio_sha256": self.digest}
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")

        prepare(source, self.output)
        [record] = json.loads((self.output / "pass-a" / "tracks.json").read_text(encoding="utf-8"))
        self.assertEqual(record["track_id"], row["track_id"])
        self.assertEqual(record["duration_seconds"], 0.5)
