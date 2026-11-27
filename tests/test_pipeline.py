import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chorustrace import (
    FeatureConfig,
    Recognizer,
    TrainingConfig,
    evaluate,
    prepare_benchmark,
    read_manifest,
    train,
)
from chorustrace.audio import read_wav, write_wav
from chorustrace.data import Example
from chorustrace.synthetic import make_synthetic_sources


class PipelineTests(unittest.TestCase):
    def test_manifest_refuses_song_across_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            row = {"audio": "a.wav", "text": "hello", "song_id": "same", "split": "train"}
            other = dict(row, audio="b.wav", split="test")
            path.write_text(f"{json.dumps(row)}\n{json.dumps(other)}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_manifest(path)

    def test_paired_mix_levels_preserve_vocal_and_original_mix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rate = 8000
            times = np.arange(rate) / rate
            vocal = 0.2 * np.sin(2 * np.pi * 440 * times)
            backing = 0.1 * np.sin(2 * np.pi * 200 * times)
            peaks = {"vocal": vocal, "accompaniment": backing, "mix": vocal + backing / 2}
            source = {"song_id": "song", "clip_id": "one", "split": "test", "text": "Hello!"}
            for name, samples in peaks.items():
                source[name] = f"{name}.wav"
                write_wav(root / source[name], samples, rate)
            sources = root / "sources.jsonl"
            sources.write_text(json.dumps(source) + "\n", encoding="utf-8")
            output = prepare_benchmark(sources, os.path.relpath(root / "prepared"), (-6, 0, 20))
            examples = read_manifest(output)
            audio = {example.condition: read_wav(example.audio)[0] for example in examples}

            np.testing.assert_allclose(audio["original_mix"], audio["vocal"] + audio["instrumental"] / 2,
                                       atol=6e-5)
            for level, condition in [(-6, "mix_-6dB"), (0, "mix_+0dB"), (20, "mix_+20dB")]:
                ratio = np.linalg.norm(audio[condition] - audio["vocal"]) / np.linalg.norm(audio["vocal"])
                self.assertAlmostEqual(20 * np.log10(ratio), level, places=2)
            self.assertEqual({example.split for example in examples}, {"test"})
            self.assertEqual([example.text for example in examples if example.condition == "instrumental"], [""])

            # A later failure must leave the prior manifest and clip files readable together.
            original_manifest = output.read_bytes()
            original_vocal = examples[0].audio.read_bytes()
            broken = dict(source, clip_id="broken", vocal="absent.wav")
            sources.write_text(f"{json.dumps(source)}\n{json.dumps(broken)}\n", encoding="utf-8")
            write_wav(root / source["vocal"], vocal / 2, rate)
            with self.assertRaises(FileNotFoundError):
                prepare_benchmark(sources, root / "prepared")
            self.assertEqual(output.read_bytes(), original_manifest)
            self.assertEqual(examples[0].audio.read_bytes(), original_vocal)

    def test_memorizes_and_reloads_one_sung_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = make_synthetic_sources(root)
            source = json.loads(sources.read_text(encoding="utf-8").splitlines()[0])
            clip = Example(root / source["vocal"], source["text"], source["song_id"])
            recognizer = train([clip], TrainingConfig(epochs=80, learning_rate=0.008,
                                                      features=FeatureConfig(hop_size=320)))
            checkpoint = root / "checkpoint.npz"
            recognizer.save(checkpoint)
            restored = Recognizer.load(checkpoint)
            self.assertEqual(clip.text, "hello moon")
            self.assertEqual(restored.transcribe(clip.audio), clip.text)
            np.testing.assert_array_equal(restored.posteriors(clip.audio), recognizer.posteriors(clip.audio))
            with self.assertRaisesRegex(ValueError, "overlap training"):
                restored.check_held_out([clip])

    def test_micro_averages_and_empty_references(self):
        class FakeRecognizer:
            def transcribe(self, audio):
                return {"one": "we no", "two": "", "three": "la", "four": ""}[audio.name]

        examples = [
            Example(Path("one"), "we go", "a", "test", "vocal"),
            Example(Path("two"), "hi", "b", "test", "vocal"),
            Example(Path("three"), "", "a", "test", "instrumental"),
            Example(Path("four"), "", "b", "test", "instrumental"),
        ]
        sections = evaluate(FakeRecognizer(), examples, samples=1)["conditions"]
        self.assertAlmostEqual(sections["vocal"]["wer"], 2 / 3)
        self.assertAlmostEqual(sections["vocal"]["cer"], 3 / 7)
        self.assertEqual(sections["instrumental"]["instrumental_with_words"], 1)
        self.assertEqual(sections["instrumental"]["instrumental_word_rate"], 0.5)
        self.assertIsNone(sections["instrumental"]["wer"])
