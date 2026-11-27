import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from chorustrace.audio import write_wav
from chorustrace.ctc import log_softmax
from chorustrace.data import Example
from chorustrace.fulltrack_data import FullTrack, TimedReferenceWord
from chorustrace.fulltrack_training import _clip_spans, _prepare_clips, _training_groups, train_full_tracks
from chorustrace.model import AcousticModel
from chorustrace.recognizer import Recognizer, TrainingConfig, train


def _track(track_id, artists, song_id=None, *, words=(), instrumental=(), duration=18.0):
    return FullTrack(
        track_id, song_id or track_id, tuple(artists), track_id, track_id,
        "pop", "train", Path("unused.wav"), f"lyric {track_id}", duration,
        tuple(words), tuple(instrumental),
    )


class FullTrackTrainingTests(unittest.TestCase):
    def test_ctc_prior_updates_from_previous_epochs_raw_frame_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            examples = []
            for index, (frames, label) in enumerate(((4000, "go"), (7500, "moon"))):
                audio = directory / f"clip-{index}.wav"
                time = np.arange(frames) / 16000
                write_wav(audio, 0.3 * np.sin(2 * np.pi * (440 + 70 * index) * time), 16000)
                examples.append(Example(audio, label, f"song-{index}"))

            observed = []
            original = AcousticModel.loss_and_gradients

            def record(model, features, targets, *, log_prior_offset, frame_probability_sum):
                raw = np.exp(log_softmax(model.forward(features)))
                observed.append((raw, log_prior_offset.copy()))
                return original(
                    model, features, targets, log_prior_offset=log_prior_offset,
                    frame_probability_sum=frame_probability_sum,
                )

            weight = 0.3
            config = TrainingConfig(epochs=2, hidden_size=4, ctc_prior_weight=weight)
            with patch.object(AcousticModel, "loss_and_gradients", record):
                recognizer = train(examples, config)
            self.assertEqual(len(observed), 4)
            initial = -weight * np.log(recognizer.model.n_classes)
            for _, offset in observed[:2]:
                np.testing.assert_allclose(offset, initial)
            prior = np.concatenate([raw for raw, _ in observed[:2]]).mean(axis=0)
            for _, offset in observed[2:]:
                np.testing.assert_allclose(offset, weight * np.log(prior), atol=1e-14)

    def test_input_deltas_checkpoint_preserves_weights_with_decoder_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio = directory / "train.wav"
            time = np.arange(5600) / 16000
            write_wav(audio, 0.3 * time * np.sin(2 * np.pi * (440 + 25 * time) * time), 16000)
            config = TrainingConfig(epochs=2, hidden_size=4, dilations=(1, 5), input_deltas=True)
            recognizer = train([Example(audio, "go", "training")], config)
            recognizer.blank_logit_offset = 0.75
            checkpoint = directory / "deltas.npz"
            recognizer.save(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"].item()))
                self.assertEqual(metadata["version"], 4)
                self.assertIs(metadata["input_deltas"], True)
            reloaded = Recognizer.load(checkpoint)
            self.assertTrue(reloaded.model.input_deltas)
            self.assertEqual(reloaded.model.dilations, (1, 5))
            self.assertEqual(reloaded.blank_logit_offset, 0.75)
            self.assertGreater(np.linalg.norm(reloaded.model.parameters["delta_weight"]), 0)
            np.testing.assert_array_equal(reloaded.posteriors(audio), recognizer.posteriors(audio))

    def test_custom_dilation_training_checkpoint_preserves_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio = directory / "train.wav"
            time = np.arange(5600) / 16000
            write_wav(audio, 0.3 * np.sin(2 * np.pi * 440 * time), 16000)
            config = TrainingConfig(epochs=2, hidden_size=4, dilations=[1, 3, 9, 27, 81])
            recognizer = train([Example(audio, "go", "training")], config)
            checkpoint = directory / "custom.npz"
            recognizer.save(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertEqual(json.loads(str(saved["metadata"].item()))["version"], 2)
            reloaded = Recognizer.load(checkpoint)
            self.assertEqual(reloaded.model.dilations, (1, 3, 9, 27, 81))
            self.assertEqual(reloaded.model.context_frames, 121)
            self.assertEqual(reloaded.training_song_ids, ("training",))
            np.testing.assert_array_equal(reloaded.posteriors(audio), recognizer.posteriors(audio))

    def test_invalid_dilation_config_is_rejected(self):
        for values in ((), (0,), (-3,), (1, 1), (True,), (1.5,), None):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "distinct positive integers"):
                TrainingConfig(dilations=values)

    def test_disk_feature_cache_preserves_training_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            examples = []
            for index, (frames, label) in enumerate(((4000, "go"), (7500, "moon"), (5200, ""))):
                audio = directory / f"clip-{index}.wav"
                time = np.arange(frames) / 16000
                write_wav(audio, 0.3 * np.sin(2 * np.pi * (440 + 70 * index) * time), 16000)
                examples.append(Example(audio, label, f"song-{index}"))

            config = TrainingConfig(epochs=3, hidden_size=4, seed=7)
            expected = train(examples, config)
            cached = train(examples, config, feature_cache_dir=directory)
            np.testing.assert_array_equal(cached.feature_mean, expected.feature_mean)
            np.testing.assert_array_equal(cached.feature_scale, expected.feature_scale)
            for name, expected_value in expected.model.parameters.items():
                np.testing.assert_array_equal(cached.model.parameters[name], expected_value)
            self.assertCountEqual(directory.iterdir(), [example.audio for example in examples])

    def test_validation_chooses_earlier_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "train.wav"
            time = np.arange(4000) / 16000
            write_wav(audio, 0.3 * np.sin(2 * np.pi * 440 * time), 16000)
            training = [Example(audio, "go", "training")]
            scores = {1: (0.25, 0.17), 2: (0.5, 0.10), 3: (0.25, 0.17)}
            expected = train(training, TrainingConfig(epochs=1, hidden_size=4))
            selected = train(
                training,
                TrainingConfig(epochs=3, hidden_size=4),
                validation_score=lambda epoch, _: scores[epoch],
            )
            last = train(training, TrainingConfig(epochs=3, hidden_size=4))
            self.assertFalse(np.array_equal(last.model.parameters["input_weight"], expected.model.parameters["input_weight"]))
            for name, expected_value in expected.model.parameters.items():
                np.testing.assert_array_equal(selected.model.parameters[name], expected_value)

    def test_linked_artists_and_alternates_stay_in_nested_groups(self):
        tracks = [
            _track("track-one", ("artist-one",)),
            _track("track-two", ("artist-one", "artist-two")),
            _track("track-three", ("artist-three",), "track-two"),
            _track("track-four", ("artist-four",)),
        ]
        groups = _training_groups(tracks, seed=14)
        self.assertEqual(groups, _training_groups(list(reversed(tracks)), seed=14))
        self.assertCountEqual(
            [set(track.track_id for track in group) for group in groups],
            [{"track-one", "track-two", "track-three"}, {"track-four"}],
        )

    def test_full_track_slices_do_not_inherit_prepared_clip_phonemes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio = directory / "training.wav"
            write_wav(audio, np.zeros(4000), 16000)
            full = replace(_track(
                "full", ("artist",), words=[TimedReferenceWord("go", 0.1, 0.2)],
                duration=0.25,
            ), audio=audio)
            prepared = replace(_track("0", ("other",), duration=0.25), audio=audio,
                               full_track=False)
            examples = _prepare_clips(full, directory / "slices", 1.0)
            examples.extend(_prepare_clips(prepared, directory / "unused", 1.0))
            self.assertEqual([example.clip_id for example in examples], [None, "0"])

    def test_clip_boundaries_keep_repeats_and_use_annotated_instrumentals(self):
        track = _track(
            "train", ("artist",), words=[
                TimedReferenceWord("go", 1.0, 1.5),
                TimedReferenceWord("now", 1.6, 2.5),
                TimedReferenceWord("go", 4.0, 4.8),
                TimedReferenceWord("home", 4.9, 5.4),
            ], instrumental=[(6.0, 18.0)],
        )
        clipped = list(_clip_spans(track, 5.0))
        self.assertEqual([text for _, _, text in clipped], ["go now", "go home", "", "", ""])
        self.assertTrue(all(end - start <= 5.0 for start, end, _ in clipped))
        self.assertLess(clipped[0][1], clipped[1][0])
        self.assertEqual(clipped[2][0], 6.0)
        self.assertEqual(clipped[-1][1], 18.0)

    def test_bounded_train_excerpt_is_used_directly_and_can_keep_vocal_condition(self):
        track = replace(_track("train", ("artist",)), full_track=False, condition="vocal")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "clips"
            self.assertEqual(
                _prepare_clips(track, output, 18.0),
                [Example(track.audio, track.text, track.song_id, "train", "vocal", track.track_id)],
            )
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "at most 17 seconds"):
                _prepare_clips(track, output, 17.0)

    def test_original_mix_excerpts_select_checkpoint_with_explicit_validation_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            time = np.arange(4000) / 16000
            training_audio = directory / "training.wav"
            validation_audio = directory / "validation.wav"
            write_wav(training_audio, 0.3 * np.sin(2 * np.pi * 440 * time), 16000)
            write_wav(validation_audio, 0.3 * np.sin(2 * np.pi * 520 * time), 16000)
            training = replace(
                _track("train", ("artist",)), audio=training_audio, text="go",
                full_track=False, duration_seconds=0.25,
            )
            validation = replace(
                _track("other", ("other",)), audio=validation_audio, text="so",
                split="validation", full_track=False, duration_seconds=0.25,
            )
            checkpoint = directory / "selected.npz"
            report = train_full_tracks(
                [training, validation], checkpoint, config=TrainingConfig(epochs=1, hidden_size=4),
            )
            self.assertEqual(report["selection"], {"epoch": 1, "training_groups": 1})
            self.assertEqual(report["validation"]["aggregate"]["reference_words"], 1)
            self.assertEqual(report["validation_scope"], {
                "condition": "original_mix", "full_tracks": 0, "excerpt_tracks": 1,
                "excerpt_track_ids": ["other"],
            })
            self.assertEqual(Recognizer.load(checkpoint).training_song_ids, ("train",))

    def test_vocal_and_test_excerpts_do_not_qualify_for_validation(self):
        training = _track("train", ("artist",))
        validation = replace(
            _track("other", ("other",)), split="validation", full_track=False, condition="vocal",
        )
        with self.assertRaisesRegex(ValueError, "validation requires original-mix audio"):
            train_full_tracks([training, validation], "unused.npz")
        with self.assertRaisesRegex(ValueError, "only train and validation tracks"):
            train_full_tracks([training, replace(validation, split="test")], "unused.npz")


if __name__ == "__main__":
    unittest.main()
