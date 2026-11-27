import argparse
import hashlib
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.compare_ctc_validation import _check_bank_isolation, _frontier, compare
from chorustrace.data import write_jsonl


@dataclass
class _Model:
    test_lock_sha256: str
    beam_width: int = 1
    blank_logit_offset: float = 0.0

    def check_held_out(self, tracks):
        pass

    def save(self, path):
        path.write_text(str(self.blank_logit_offset), encoding="utf-8")


class JointValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest, self.lock, self.bank = (self.root / name for name in ("dev", "lock", "bank"))
        for path in (self.manifest, self.lock, self.bank):
            path.write_bytes(path.name.encode())
        self.output = self.root / "selected.npz"
        self.acoustic = _Model(hashlib.sha256(self.lock.read_bytes()).hexdigest())

    def test_frontier_retains_tradeoffs_and_exact_ties(self):
        def point(wer, cer, alarm):
            return {"wer": wer, "cer": cer, "instrumental": {"words_per_minute": alarm}}

        a = point(0.3, 0.4, 8.0)
        b = point(0.4, 0.3, 2.0)
        c = point(0.4, 0.3, 2.0)
        dominated = point(0.4, 0.3, 3.0)
        self.assertEqual(_frontier([a, b, c, dominated]), [a, b, c])

    def _compare(self, cap):
        args = argparse.Namespace(
            manifest=self.manifest, lock=self.lock, instrumental=self.bank,
            output=self.output, candidates=[self.root / "candidate.npz"],
            joint_instrumental=True, max_instrumental_words_per_minute=cap,
        )
        validation = SimpleNamespace(split="validation", genre="rock", diagnostic_only=False)
        diagnostic = SimpleNamespace(split="validation", genre="pop", diagnostic_only=True)

        def evaluate(model, tracks, *, samples):
            self.assertEqual(tracks, [validation, diagnostic])
            edits = 2 if model.blank_logit_offset else 3
            alarms = 4 if model.blank_logit_offset else 1
            return {
                "tracks": [{"genre": "rock", "reference_words": 10, "reference_characters": 70,
                            "word_edits": edits, "deletions": edits, "substitutions": 0, "insertions": 0,
                            "character_edits": edits * 2}],
                "instrumental_diagnostics": {"instrumental": {"words_per_minute": alarms, "words": alarms,
                                                               "seconds": 60.0}},
            }

        module = "benchmarks.compare_ctc_validation"
        with (patch(f"{module}._DECODERS", ((1, 1.0),)),
              patch(f"{module}.Recognizer.load", return_value=self.acoustic),
              patch(f"{module}.read_fulltrack_development", return_value=[validation]),
              patch(f"{module}.read_bank", return_value=[diagnostic]),
              patch(f"{module}._check_bank_isolation"),
              patch(f"{module}.evaluate_full_tracks", side_effect=evaluate)):
            return compare(args)

    def test_cap_selects_feasible_setting_over_lower_wer_with_higher_false_alarms(self):
        report = self._compare(1.0)
        self.assertEqual(report["selection"]["wer"], 0.3)
        self.assertEqual(report["selection"]["instrumental"]["words_per_minute"], 1)
        self.assertEqual(len(report["joint_validation"]["frontier"]), 2)
        self.assertEqual(report["joint_validation"]["eligible_configurations"], 1)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "0.0")

    def test_cap_with_no_eligible_configuration_reports_frontier_without_checkpoint(self):
        report = self._compare(0.0)
        self.assertIsNone(report["selection"])
        self.assertIsNone(report["checkpoint_sha256"])
        self.assertEqual(len(report["joint_validation"]["frontier"]), 2)
        self.assertFalse(self.output.exists())

    def test_bank_checks_individual_performer_credits_against_protected_splits(self):
        write_jsonl(self.manifest, [{"artist_ids": ["Artist One & Artist Two"], "split": "train"}])
        write_jsonl(self.bank, [{"artist_ids": ["Artist Two"], "split": "validation"}])
        with patch("benchmarks.compare_ctc_validation._read_lock", return_value=[]):
            with self.assertRaisesRegex(ValueError, "instrumental bank overlaps"):
                _check_bank_isolation(self.bank, self.manifest, self.lock)
