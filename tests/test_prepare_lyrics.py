import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chorustrace.audio import read_wav, write_wav
from chorustrace.data import read_jsonl, write_jsonl
from scripts.prepare_lyrics import freeze, verify_frozen


class PrepareLyricsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / "sources.jsonl"
        self.rights = self.root / "rights.jsonl"
        self.permissions = {
            "rights_id": "fixture",
            "source": "procedural test fixture",
            "recording_permission": "generated for this test",
            "lyrics_permission": "generated for this test",
            "training_allowed": True,
            "checkpoint_release_allowed": True,
        }
        write_jsonl(self.rights, [self.permissions])
        self.rng = np.random.default_rng(5)
        self.rate = 8000

    def clip(self, identifier, song=None, artist=None, paired=False):
        vocal = 0.07 * self.rng.normal(size=self.rate)
        row = {
            "song_id": song or identifier,
            "recording_id": identifier,
            "clip_id": identifier,
            "artist": artist or identifier,
            "genre": "fixture",
            "rights_id": "fixture",
            "language": "en",
            "annotation_verified": True,
            "text": "New Moon!",
            "start": 0.1,
            "end": 0.8,
        }
        tracks = {"vocal": vocal}
        if paired:
            backing = 0.04 * self.rng.normal(size=self.rate)
            tracks.update(accompaniment=backing, mix=vocal + backing)
            row["alignment_verified"] = True
        for name, samples in tracks.items():
            filename = f"{identifier}-{name}.wav"
            write_wav(self.root / filename, samples, self.rate)
            row[name] = filename
        return row

    def prepare(self, rows, output="prepared"):
        write_jsonl(self.sources, rows)
        return freeze(self.sources, self.rights, self.root / output, seed=11)

    def test_pairs_are_deterministic_and_artist_song_groups_stay_together(self):
        rows = [
            self.clip("original", song="a-song", artist="Vocalist A", paired=True),
            self.clip("unrelated-by-song", artist="vocalist a"),
            self.clip("cover", song="a-song", artist="vocalist b"),
            self.clip("cover-artist-song", artist="vocalist b"),
            self.clip("solo-c"),
            self.clip("solo-d"),
        ]
        manifest = self.prepare(rows)
        repeated = self.prepare(list(reversed(rows)), "repeated")
        self.assertEqual(manifest.read_bytes(), repeated.read_bytes())
        lock = json.loads((manifest.parent / "freeze.json").read_text())
        repeated_lock = json.loads((repeated.parent / "freeze.json").read_text())
        self.assertEqual(lock["audio_sha256"], repeated_lock["audio_sha256"])
        self.assertEqual(verify_frozen(manifest.parent), manifest)

        prepared = [row for _, row in read_jsonl(manifest)]
        paired = {row["condition"]: row for row in prepared if row["clip_id"] == "original"}
        self.assertEqual(len(paired), 7)
        self.assertEqual(paired["instrumental"]["text"], "")
        self.assertEqual(paired["vocal"]["text"], "new moon")
        self.assertEqual(paired["vocal"]["duration_seconds"], 0.7)
        waves = {key: read_wav(manifest.parent / row["audio"])[0] for key, row in paired.items()}
        np.testing.assert_allclose(waves["original_mix"], waves["vocal"] + waves["instrumental"], atol=6e-5)
        for level in (-12, -6, 0, 6):
            mix = waves[f"mix_{level:+g}dB"]
            ratio = np.linalg.norm(mix - waves["vocal"]) / np.linalg.norm(waves["vocal"])
            self.assertAlmostEqual(20 * np.log10(ratio), level, places=2)
        connected = {"original", "unrelated-by-song", "cover", "cover-artist-song"}
        splits = {row["split"] for row in prepared if row["clip_id"] in connected}
        self.assertEqual(len(splits), 1)
        self.assertEqual({row["split"] for row in prepared}, {"train", "validation", "test"})
        self.assertEqual({row["condition"] for row in prepared if row["clip_id"] == "solo-c"}, {"vocal"})

    def test_freeze_refuses_overwrite_and_detects_changed_wav_and_manifest(self):
        manifest = self.prepare([self.clip("solo")])
        self.assertTrue(verify_frozen(manifest.parent).is_file())
        with self.assertRaises(FileExistsError):
            freeze(self.sources, self.rights, manifest.parent)
        _, row = next(read_jsonl(manifest))
        audio = manifest.parent / row["audio"]
        original = audio.read_bytes()
        audio.write_bytes(original + b"invalid")
        with self.assertRaisesRegex(ValueError, "audio does not match"):
            verify_frozen(manifest.parent)
        audio.write_bytes(original)
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            verify_frozen(manifest.parent)
        (manifest.parent / "freeze.json").write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(TypeError, "expected JSON object"):
            verify_frozen(manifest.parent)

    def test_rejects_unconfirmed_rights_bounds_annotation_or_alignment(self):
        row = self.clip("paired", paired=True)
        write_jsonl(self.rights, [dict(self.permissions, training_allowed=False)])
        with self.assertRaisesRegex(ValueError, "no affirmative training permission"):
            self.prepare([row])
        write_jsonl(self.rights, [self.permissions])
        with self.assertRaisesRegex(ValueError, "reviewed single-lead English"):
            self.prepare([dict(row, annotation_verified=False)])
        with self.assertRaisesRegex(ValueError, "bounds exceed"):
            self.prepare([dict(row, end=1.1)])
        mix_file = self.root / row["mix"]
        mix, rate = read_wav(mix_file)
        write_wav(mix_file, np.roll(mix, 80), rate)
        with self.assertRaisesRegex(ValueError, "zero lag"):
            self.prepare([row])
        self.assertFalse((self.root / "prepared").exists())

    def test_rejects_cross_split_artists_and_duplicate_audio(self):
        first = dict(self.clip("first", artist="Singer 1"), split="train")
        second = dict(self.clip("second", artist="SINGER  1"), split="test")
        with self.assertRaisesRegex(ValueError, "artists and all versions"):
            self.prepare([first, second])
        second["artist"] = "Singer 2"
        (self.root / second["vocal"]).write_bytes((self.root / first["vocal"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "identical WAV occurs in multiple splits"):
            self.prepare([first, second])

    def test_rejects_overlapping_source_intervals(self):
        row = self.clip("first")
        overlapping = dict(row, clip_id="second", start=0.6, end=0.9)
        with self.assertRaisesRegex(ValueError, "overlapping excerpts"):
            self.prepare([row, overlapping])

    def test_original_only_has_no_synthesized_conditions(self):
        row = self.clip("original")
        row["mix"] = row.pop("vocal")
        manifest = self.prepare([row])
        self.assertEqual(
            [(item["split"], item["condition"]) for _, item in read_jsonl(manifest)],
            [("train", "original_mix")],
        )


if __name__ == "__main__":
    unittest.main()
