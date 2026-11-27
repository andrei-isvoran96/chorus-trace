import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chorustrace.audio import write_wav
from chorustrace.data import write_jsonl


class FullTrackCliTests(unittest.TestCase):
    def test_checkpoint_reloads_against_its_frozen_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            rate = 16000
            time = np.arange(rate // 2) / rate
            for index in range(23):
                track_id = f"fixture-{index}"
                audio = root / f"{track_id}.wav"
                reference = root / f"{track_id}.json"
                word = "word" + chr(ord("a") + index)
                signal = 0.1 * np.sin(2 * np.pi * (200 + 15 * index) * time)
                samples = np.column_stack((signal, signal / 2)) if index % 2 == 0 else signal
                write_wav(audio, samples, rate)
                reference.write_text(json.dumps({
                    "text": word, "complete_lead_vocal": True,
                    "transcribed_by": "fixture-author", "checked_by": "fixture-reviewer",
                    "review_evidence": "generated label",
                    "words": [{"word": word, "start": 0.1, "end": 0.4}],
                    "instrumental_intervals": [[0, 0.1], [0.4, 0.5]],
                }), encoding="utf-8")
                rows.append({
                    "track_id": track_id, "song_id": track_id, "artist_ids": [track_id],
                    "lyric_id": track_id, "recording_family_id": track_id,
                    "genre": ("pop", "rock", "hip-hop", "jazz")[index % 4],
                    "split": ("train", "validation")[index] if index < 2 else "test",
                    "audio": audio.name, "reference": reference.name,
                    "condition": "original_mix", "language": "en", "full_track": True,
                    "audio_source": "generated test fixture", "source_frames": len(time),
                    "source_sample_rate": rate, "recording_permission": "fixture author",
                    "lyrics_permission": "fixture author",
                })

            rows[-1]["diagnostic_only"] = True
            (root / rows[-1]["reference"]).write_text(json.dumps({
                "checked_by": "fixture-reviewer", "review_evidence": "instrumental fixture",
                "instrumental_intervals": [[0, 0.5]],
            }), encoding="utf-8")
            development, test = root / "development.jsonl", root / "test.jsonl"
            lock, checkpoint = root / "test.lock.json", root / "selected.npz"
            write_jsonl(development, rows[:2])
            write_jsonl(test, rows[2:])
            self.cli("freeze-test", test, lock)
            frozen = json.loads(self.cli("verify-test", test, lock).stdout)
            training = json.loads(self.cli("train-full", development, lock, checkpoint,
                                           "--epochs", "1", "--hidden-size", "4",
                                           "--stereo-features",
                                           "--dilations", "1", "3", "9", "27").stdout)
            result = json.loads(self.cli("evaluate-full", checkpoint, test, lock).stdout)
            self.assertEqual(result["aggregate"]["reference_words"], 20)
            self.assertEqual(result["track_count"], frozen["tracks"])
            self.assertEqual(frozen["instrumental_diagnostic_tracks"], 1)
            self.assertEqual(result["instrumental_diagnostics"]["track_count"], 1)
            self.assertEqual(result["test_lock_sha256"], frozen["test_lock_sha256"])
            self.assertEqual(result["checkpoint_sha256"], training["checkpoint_sha256"])
            self.assertTrue(training["configuration"]["features"]["stereo_features"])
            self.assertFalse(result["meets_goal"])
            sample = result["samples"][0]
            audio = root / f"{sample['track_id']}.wav"
            self.assertEqual(self.cli("transcribe", checkpoint, audio).stdout.rstrip("\n"), sample["prediction"])

            lock.write_text(lock.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            error = self.cli("evaluate-full", checkpoint, test, lock, check=False)
            self.assertEqual(error.returncode, 1)
            self.assertIn("checkpoint was not trained with this frozen test lock", error.stderr)

    def cli(self, *arguments, check=True):
        return subprocess.run(
            [sys.executable, "-m", "chorustrace", *map(str, arguments)],
            cwd=Path(__file__).resolve().parents[1],
            env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"},
            check=check, capture_output=True, text=True,
        )
