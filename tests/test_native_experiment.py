import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.run_native_experiment import _parser, run
from chorustrace.audio import write_wav
from chorustrace.data import write_jsonl
from chorustrace.recognizer import Recognizer


class NativeExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train_row = self._row("one", "train", "red", 330)
        self.validation_row = self._row("two", "validation", "green green", 440)
        self.manifest = self.root / "development.jsonl"
        write_jsonl(self.manifest, [self.train_row, self.validation_row])

    def _row(self, name, split, text, frequency):
        time = np.arange(8000) / 16000
        write_wav(self.root / f"{name}.wav", np.sin(2 * np.pi * frequency * time) / 2, 16000)
        reference_path = self.root / f"{name}.json"
        reference_path.write_text(json.dumps({
            "complete_lead_vocal": True,
            "transcribed_by": "generated fixture",
            "text": text,
        }), encoding="utf-8")
        return {
            "track_id": name,
            "song_id": f"{name}-song",
            "artist_ids": [f"{name}-artist"],
            "lyric_id": f"{name}-lyrics",
            "recording_family_id": f"{name}-recording",
            "genre": "pop",
            "split": split,
            "audio": f"{name}.wav",
            "reference": reference_path.name,
            "condition": "original_mix",
            "full_track": False,
            "language": "en",
            "audio_source": "test fixture",
            "source_frames": 8000,
            "source_sample_rate": 16000,
            "recording_permission": "generated",
            "lyrics_permission": "generated",
        }

    def _run(self, command, manifest, checkpoint, report, *options):
        argv = [command, str(manifest), str(checkpoint), str(report), *options]
        with contextlib.redirect_stderr(io.StringIO()):
            return run(_parser().parse_args(argv), argv)

    def _train(self, encoder):
        checkpoint = self.root / f"{encoder}.npz"
        report = self.root / f"{encoder}.json"
        result = self._run(
            "train", self.manifest, checkpoint, report,
            "--epochs", "1", "--hidden-size", "4", "--seed", "11", "--encoder", encoder,
        )
        self.assertEqual(json.loads(report.read_text())["seed"], 11)
        self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual(result["prepared_training_audio_hours"], 0.5 / 3600)
        self.assertEqual(result["labeled_training_audio_hours_by_condition"], {"original_mix": 0.5 / 3600})
        return checkpoint, result

    def test_train_encoders_use_same_data_and_reload_through_normal_inference(self):
        baseline_checkpoint, baseline = self._train("baseline")
        candidate_checkpoint, candidate = self._train("wide-context")
        for field in ("train_inventory_sha256", "validation_inventory_sha256", "manifest_sha256"):
            self.assertEqual(baseline[field], candidate[field])
        baseline_config, candidate_config = baseline["configuration"], candidate["configuration"]
        self.assertEqual({key for key in baseline_config if baseline_config[key] != candidate_config[key]}, {"dilations"})
        base_model = Recognizer.load(baseline_checkpoint).model
        candidate_recognizer = Recognizer.load(candidate_checkpoint)
        self.assertGreater(candidate_recognizer.model.context_frames, base_model.context_frames)
        self.assertEqual(
            sum(value.size for value in candidate_recognizer.model.parameters.values()),
            sum(value.size for value in base_model.parameters.values()),
        )
        self.assertIsInstance(candidate_recognizer.transcribe(self.root / self.validation_row["audio"]), str)
        verified = self._run("score", self.manifest, candidate_checkpoint, self.root / "verify.json")
        self.assertEqual(verified["checkpoint_sha256"], candidate["checkpoint_sha256"])
        self.assertEqual(verified["validation_inventory_sha256"], candidate["validation_inventory_sha256"])
        self.assertEqual(verified["validation"]["aggregate"], candidate["validation"]["aggregate"])

    def test_paired_scores_use_identical_references_and_require_matched_durations(self):
        checkpoint, _ = self._train("baseline")
        paired = self.root / "paired.jsonl"
        report = self.root / "paired.json"
        paired_row = self.validation_row | {
            "vocal_audio": self.train_row["audio"],
            "vocal_audio_source": "same-window fixture",
            "vocal_recording_permission": "generated",
        }
        write_jsonl(paired, [paired_row])
        result = self._run("paired", paired, checkpoint, report)
        for scores in result["conditions"].values():
            aggregate = scores["aggregate"]
            self.assertEqual(aggregate["reference_words"], 2)
            self.assertEqual(aggregate["wer"], (
                aggregate["substitutions"] + aggregate["deletions"] + aggregate["insertions"]
            ) / 2)
            self.assertEqual(scores["samples"], [])
        write_wav(self.root / self.train_row["audio"], np.zeros(4000), 16000)
        write_jsonl(paired, [paired_row | {"vocal_source_frames": 4000}])
        with self.assertRaisesRegex(ValueError, "cover different durations"):
            self._run("paired", paired, checkpoint, report)

    def test_refuses_test_rows_before_opening_assets(self):
        write_jsonl(self.manifest, [self.train_row, self.validation_row, {"split": "test"}])
        with self.assertRaisesRegex(ValueError, "require exactly these development splits"):
            self._run("train", self.manifest, self.root / "checkpoint.npz", self.root / "report.json")


if __name__ == "__main__":
    unittest.main()
