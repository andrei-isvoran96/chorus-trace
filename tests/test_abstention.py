import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from chorustrace.abstention import abstain, decode_with_confidence, load_threshold
from chorustrace.data import ALPHABET, CHARACTER_IDS, file_sha256, write_jsonl
from scripts.abstention_experiment import calibrate, select_threshold


def frame(character, peak):
    label = CHARACTER_IDS[character] if character is not None else 0
    probabilities = np.full(len(ALPHABET) + 1, (1 - peak) / len(ALPHABET))
    probabilities[label] = peak
    return probabilities


class AbstentionTests(unittest.TestCase):
    def test_confidence_counts_separate_ctc_runs_and_ignores_spaces(self):
        probabilities = np.array([
            frame("a", 0.4), frame("a", 0.7), frame(None, 0.99),
            frame("a", 0.5), frame(" ", 0.99), frame("b", 0.6),
        ])
        text, confidence = decode_with_confidence(probabilities)
        self.assertEqual(text, "aa b")
        self.assertAlmostEqual(confidence, 0.6)
        self.assertEqual(decode_with_confidence(np.array([frame(" ", 0.99)])), ("", 0.0))
        self.assertEqual(abstain(text, confidence, 0.59), text)
        self.assertEqual(abstain(text, confidence, confidence), "")

    def test_calibration_limit_is_strict_and_ties_are_suppressed(self):
        self.assertEqual(select_threshold([0.7] + [0.0] * 19, 0.05), 0.7)
        self.assertEqual(select_threshold([0.7] + [0.0] * 20, 0.05), 0.0)
        self.assertEqual(select_threshold([0.7, 0.7] + [0.0] * 19, 0.05), 0.7)
        with self.assertRaisesRegex(ValueError, "validation instrumentals"):
            select_threshold([], 0.05)

    def test_rejects_non_object_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "calibration.json"
            for invalid in ([], 0.2, None):
                sidecar.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unsupported abstention calibration"):
                    load_threshold(sidecar, "unused-model.npz")

    def test_only_paired_validation_rows_fit_the_checkpoint_bound_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.npz"
            checkpoint.write_bytes(b"fixture checkpoint")
            manifest = root / "manifest.jsonl"
            rows = [
                {"audio": "train.wav", "split": "train", "song_id": "training", "clip_id": "line",
                 "text": "a", "condition": "vocal"},
                {"audio": "test.wav", "split": "test", "song_id": "test", "clip_id": "line",
                 "text": "a", "condition": "vocal"},
            ]
            for condition, text in [("vocal", "a"), ("original_mix", "a"), ("instrumental", "")]:
                rows.append({"audio": f"{condition}.wav", "split": "validation", "song_id": "validation",
                             "clip_id": "line", "text": text, "condition": condition})
            rows.append(dict(rows[-2], audio="unpaired.wav", clip_id="unpaired"))
            write_jsonl(manifest, rows)
            posteriors = {
                "vocal.wav": np.array([frame("a", 0.9)]),
                "original_mix.wav": np.array([frame("a", 0.5)]),
                "instrumental.wav": np.array([frame("a", 0.7)]),
            }
            recognizer = Mock()
            recognizer.posteriors.side_effect = lambda audio: posteriors[audio.name]
            with patch("scripts.abstention_experiment.Recognizer.load", return_value=recognizer):
                result = calibrate(checkpoint, manifest, 0.05, fixture=True)
                with patch("scripts.prepare_lyrics.verify_frozen", return_value=manifest):
                    recognizer.training_manifest_sha256 = "different"
                    with self.assertRaisesRegex(ValueError, "frozen manifest"):
                        calibrate(checkpoint, manifest, 0.05)

            self.assertEqual(recognizer.posteriors.call_count, 3)
            self.assertEqual(len(recognizer.check_held_out.call_args.args[0]), 4)
            self.assertEqual(result["validation_excerpts"], 2)
            self.assertEqual(result["paired_validation_excerpts"], 1)
            self.assertEqual(result["threshold"], 0.7)
            conditions = result["conditions"]
            self.assertEqual(conditions["instrumental"]["before"]["instrumental_false_alarms"], 1)
            self.assertEqual(conditions["instrumental"]["after"]["instrumental_false_alarms"], 0)
            self.assertEqual(conditions["original_mix"]["before"]["wer"], 0.0)
            self.assertEqual(conditions["original_mix"]["after"]["wer"], 1.0)
            self.assertEqual(conditions["original_mix"]["after"]["voiced_rejects"], 1)
            self.assertEqual(conditions["original_mix"]["matched_vocal"]["after"]["voiced_rejects"], 0)

            sidecar = root / "calibration.json"
            sidecar.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(load_threshold(sidecar, checkpoint, manifest_sha256=file_sha256(manifest)), 0.7)
            with self.assertRaisesRegex(ValueError, "different manifest"):
                load_threshold(sidecar, checkpoint, manifest_sha256="other")
            checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                load_threshold(sidecar, checkpoint)
