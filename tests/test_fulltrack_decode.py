import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from chorustrace.audio import FeatureConfig, write_wav
from chorustrace.ctc import greedy_decode
from chorustrace.data import ALPHABET, CHARACTER_IDS, decode_characters
from chorustrace.model import AcousticModel
from chorustrace.recognizer import Recognizer


class FullTrackDecodeTests(unittest.TestCase):
    def test_checkpointed_beam_carries_alignment_across_segments(self):
        config = FeatureConfig()
        model = AcousticModel(config.n_mels, len(ALPHABET) + 1, 5, input_deltas=True)
        recognizer = Recognizer(model, config, np.zeros(config.n_mels), np.ones(config.n_mels),
                               blank_logit_offset=0.7, beam_width=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio, checkpoint = root / "example.wav", root / "model.npz"
            sound = np.random.default_rng(7).normal(0, 0.08, config.sample_rate // 2)
            write_wav(audio, sound, config.sample_rate)
            recognizer.save(checkpoint)
            restored = Recognizer.load(checkpoint)
            self.assertTrue(restored.model.input_deltas)
            self.assertEqual(restored.beam_width, 4)
            self.assertEqual(restored.blank_logit_offset, 0.7)
            whole = recognizer.transcribe_full_track(audio, segment_seconds=None)
            segmented = restored.transcribe_full_track(audio, segment_seconds=0.041)
            self.assertTrue(segmented.segment_boundaries_seconds)
            self.assertEqual((segmented.text, segmented.words), (whole.text, whole.words))

    def test_active_input_deltas_preserve_timed_words_at_segment_joins(self):
        config = FeatureConfig()
        model = AcousticModel(
            config.n_mels, len(ALPHABET) + 1, hidden_size=8, seed=9,
            dilations=(1, 5, 18), input_deltas=True,
        )
        recognizer = Recognizer(model, config, np.zeros(config.n_mels), np.ones(config.n_mels))
        rng = np.random.default_rng(9)
        model.parameters["delta_weight"][:] = rng.normal(scale=3, size=(config.n_mels, model.hidden_size))
        samples = rng.normal(scale=0.1, size=25231)
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "full.wav"
            write_wav(wav, samples, 16000)
            whole = recognizer.transcribe_full_track(wav, segment_seconds=None)
            segmented = recognizer.transcribe_full_track(wav, segment_seconds=0.17)
        self.assertTrue(whole.words)
        self.assertEqual(model.context_frames, 26)
        self.assertEqual(segmented.words, whole.words)
        np.testing.assert_allclose(segmented.segment_boundaries_seconds, np.arange(1, 10) * 0.17)

    def test_blank_offset_round_trips_and_adjusts_both_decode_paths(self):
        config = FeatureConfig()
        model = AcousticModel(config.n_mels, len(ALPHABET) + 1, hidden_size=2, dilations=(2, 5))
        for parameter in model.parameters.values():
            parameter.fill(0)
        bias = model.parameters["output_bias"]
        bias.fill(-3)
        bias[0] = 0.2
        bias[CHARACTER_IDS["a"]] = 0
        original = Recognizer(model, config, np.zeros(config.n_mels), np.ones(config.n_mels))
        adjusted = replace(original, blank_logit_offset=0.25)

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "full.wav"
            write_wav(wav, np.zeros(4000), config.sample_rate)
            checkpoint = Path(tmp) / "dilated.npz"
            original.save(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertNotIn("blank_logit_offset", json.loads(saved["metadata"].item()))
            self.assertEqual(Recognizer.load(checkpoint).blank_logit_offset, 0.0)
            adjusted.save(checkpoint)
            restored = Recognizer.load(checkpoint)
            segmented = restored.transcribe_full_track(wav, segment_seconds=0.05)
            whole = restored.transcribe_full_track(wav, segment_seconds=None)

            self.assertEqual(restored.blank_logit_offset, 0.25)
            self.assertEqual(restored.model.dilations, (2, 5))
            self.assertEqual(original.transcribe(wav), "")
            self.assertEqual(segmented.text, "a")
            self.assertEqual(segmented.words, whole.words)
            np.testing.assert_array_equal(restored.posteriors(wav), original.posteriors(wav))

        for invalid in (-0.25, float("nan"), float("inf"), "0.25", True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                replace(original, blank_logit_offset=invalid)

    def test_custom_context_larger_than_segment_matches_whole_audio(self):
        config = FeatureConfig()
        model = AcousticModel(
            config.n_mels, len(ALPHABET) + 1, hidden_size=8, seed=14,
            dilations=(1, 3, 9, 27, 81, 243),
        )
        recognizer = Recognizer(model, config, np.zeros(config.n_mels), np.ones(config.n_mels))
        samples = np.random.default_rng(3).normal(scale=0.1, size=151123)
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "full.wav"
            write_wav(wav, samples, 16000)
            segmented = recognizer.transcribe_full_track(wav, segment_seconds=1.5)
            whole = recognizer.transcribe_full_track(wav, segment_seconds=None)
        self.assertEqual(model.context_frames, 364)
        self.assertEqual(len(segmented.segment_boundaries_seconds), 6)
        self.assertTrue(whole.words)
        self.assertEqual(segmented.words, whole.words)
        self.assertEqual(segmented.text, whole.text)

    def test_segmented_decode_matches_whole_recording_with_real_model(self):
        config = FeatureConfig()
        model = AcousticModel(config.n_mels, len(ALPHABET) + 1, hidden_size=8, seed=14)
        recognizer = Recognizer(model, config, np.zeros(config.n_mels), np.ones(config.n_mels))
        rng = np.random.default_rng(3)
        samples = rng.normal(scale=0.1, size=(35053, 2))
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "full.wav"
            write_wav(wav, samples, 22050)
            segmented = recognizer.transcribe_full_track(wav, segment_seconds=0.37)
            whole = recognizer.transcribe_full_track(wav, segment_seconds=None)
            legacy = decode_characters(greedy_decode(recognizer.posteriors(wav)))

        self.assertTrue(legacy)
        self.assertEqual(legacy, segmented.text)
        self.assertEqual(whole.text, segmented.text)
        self.assertEqual(whole.words, segmented.words)
        self.assertEqual(whole.segment_boundaries_seconds, ())
        np.testing.assert_allclose(segmented.segment_boundaries_seconds, [0.37, 0.74, 1.11, 1.48])
        self.assertLessEqual(segmented.words[-1].end_seconds, len(samples) / 22050)

    def test_stitch_keeps_one_character_at_join_and_two_occurrences_of_word(self):
        frame_characters = ("", "l", "a", "a", "a", "a", " ", "", "l", "a", "a", "a",
                            "", "", " ", "", "b", "'", "", "'", "c", "c", "", "")
        config = FeatureConfig()
        features = np.zeros((len(frame_characters), config.n_mels))
        features[:, 0] = [CHARACTER_IDS.get(char, 0) for char in frame_characters]

        class LabelModel:
            context_frames = 2

            def __init__(self):
                self.windows = []

            def forward(self, window):
                self.windows.append(len(window))
                return np.eye(len(ALPHABET) + 1)[window[:, 0].astype(int)]

        model = LabelModel()
        recognizer = Recognizer(model, config, np.zeros(config.n_mels), np.ones(config.n_mels))
        with patch("chorustrace.recognizer.log_mel", return_value=features):
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "full.wav"
                write_wav(wav, np.zeros(4000), config.sample_rate)
                result = recognizer.transcribe_full_track(wav, segment_seconds=0.05)

        self.assertEqual(model.windows, [7, 9, 9, 9, 6])
        self.assertEqual(result.text, "la la b c")
        self.assertEqual([word.text for word in result.words], ["la", "la", "b", "c"])
        np.testing.assert_allclose(result.segment_boundaries_seconds, [0.05, 0.10, 0.15, 0.20])
        np.testing.assert_allclose(
            [(word.start_seconds, word.end_seconds) for word in result.words],
            [(0.01, 0.075), (0.08, 0.135), (0.16, 0.185), (0.20, 0.235)],
        )


if __name__ == "__main__":
    unittest.main()
