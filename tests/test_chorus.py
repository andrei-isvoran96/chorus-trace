import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chorustrace import align_posteriors, evaluate_choruses, pool_posteriors
from chorustrace.ctc import greedy_decode
from chorustrace.data import ALPHABET, CHARACTER_IDS, Example


class ChorusTests(unittest.TestCase):
    def test_alignment_compresses_a_repeated_frame(self):
        first = np.eye(3)[[0, 1, 0, 2, 0]]
        second = np.eye(3)[[0, 1, 1, 0, 2, 0]]
        expected_path = [[0, 0], [1, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
        np.testing.assert_array_equal(align_posteriors(first, second), expected_path)
        pooled_first, pooled_second = pool_posteriors(first, second)
        np.testing.assert_array_equal(pooled_first, first)
        np.testing.assert_array_equal(pooled_second, second)
        self.assertEqual(greedy_decode(pooled_first), [1, 2])
        self.assertEqual(greedy_decode(pooled_second), [1, 2])

    def test_a_confident_take_can_erase_a_changed_lyric(self):
        a, h, o = (CHARACTER_IDS[letter] for letter in "aho")
        first = np.eye(len(ALPHABET) + 1)[[0, a, 0, h, 0]]
        second = first.copy()
        second[1, [0, a, o]] = [0.05, 0.35, 0.60]

        class FakeRecognizer:
            def posteriors(self, audio):
                return first if audio.name == "one.wav" else second

        examples = [
            Example(Path("one.wav"), "ah", "song", "test", "original_mix", "one"),
            Example(Path("two.wav"), "oh", "song", "test", "original_mix", "two"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            pairs = Path(temporary) / "pairs.jsonl"
            pair = {"song_id": "song", "clip_a": "one", "clip_b": "two", "changed_lyric": "oh"}
            pairs.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            report = evaluate_choruses(FakeRecognizer(), examples, pairs)["pairs"][0]
        self.assertEqual(report["takes"][1]["independent"]["prediction"], "oh")
        self.assertEqual(report["takes"][1]["pooled"]["prediction"], "ah")
        self.assertTrue(report["erased_changed_lyric"])
