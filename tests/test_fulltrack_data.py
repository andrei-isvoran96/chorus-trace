import json
import tempfile
import unittest
import wave
from pathlib import Path

from chorustrace.fulltrack_data import (
    TimedReferenceWord,
    freeze_fulltrack_test,
    read_fulltrack_development,
    read_fulltrack_manifest,
    verify_fulltrack_test,
)


class FullTrackDataTests(unittest.TestCase):
    def test_training_instrumental_requires_checked_empty_intervals(self):
        row = self._track(20, "train") | {"condition": "instrumental"}
        row.pop("lyrics_permission")
        row.pop("language")
        ref = self.root / row["reference"]
        ref.write_text(json.dumps({"checked_by": "publisher", "review_evidence": "instrumental score",
                                   "instrumental_intervals": [[0, 1]]}), encoding="utf-8")
        self._write([row])
        track, = read_fulltrack_manifest(self.manifest)
        self.assertEqual(track.text, "")
        self.assertEqual(track.instrumental_intervals, ((0.0, 1.0),))
        with self.assertRaises(ValueError):
            self._write([row | {"split": "validation"}])
            read_fulltrack_manifest(self.manifest)
        ref.write_text(json.dumps({"checked_by": "publisher", "review_evidence": "instrumental score",
                                   "instrumental_intervals": [[0, 1]], "text": "hidden words"}), encoding="utf-8")
        self._write([row])
        with self.assertRaisesRegex(ValueError, "cannot contain lyric"):
            read_fulltrack_manifest(self.manifest)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.jsonl"
        self.lock = self.root / "test-lock.json"

    def _track(self, number, split="test"):
        # Unique audio and lyric bytes in short fixtures without shipping music.
        name = f"track-{number}"
        with wave.open(str(self.root / f"{name}.wav"), "wb") as stream:
            stream.setparams((1, 1, 8000, 8000, "NONE", "not compressed"))
            stream.writeframes(bytes([number]) * 8000)
        reference = {
            "text": f"Oh {chr(97 + number)} oh!",
            "complete_lead_vocal": True,
            "transcribed_by": "transcriber",
            "checked_by": "reviewer",
            "review_evidence": "fixture review log",
            "instrumental_intervals": [[0, 0.2]],
            "words": [
                {"word": "Oh", "start": 0.2, "end": 0.4},
                {"word": chr(97 + number), "start": 0.5, "end": 0.6},
                {"word": "oh", "start": 0.7, "end": 0.9},
            ],
        }
        (self.root / f"{name}.json").write_text(json.dumps(reference), encoding="utf-8")
        return {
            "track_id": name,
            "song_id": f"work-{number}",
            "artist_ids": [f"artist-{number}"],
            "lyric_id": f"lyrics-{number}",
            "recording_family_id": f"versions-{number}",
            "genre": ("pop", "rock", "hip hop", "folk")[number % 4],
            "split": split,
            "audio": f"{name}.wav",
            "reference": f"{name}.json",
            "condition": "original_mix",
            "language": "en",
            "full_track": True,
            "audio_source": "fixture source record",
            "source_frames": 8000,
            "source_sample_rate": 8000,
            "recording_permission": "fixture recording grant",
            "lyrics_permission": "fixture lyric grant",
        }

    def _write(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _diagnostic(self, number, **extra):
        row = self._track(number) | {"diagnostic_only": True} | extra
        del row["lyrics_permission"]
        del row["language"]
        (self.root / row["reference"]).write_text(json.dumps({
            "checked_by": "published review",
            "review_evidence": "fixture instrumentation annotation",
            "instrumental_intervals": [[0.15, 0.65]],
        }), encoding="utf-8")
        return row

    def _freeze(self):
        rows = [self._track(index) for index in range(20)]
        self._write(rows)
        freeze_fulltrack_test(self.manifest, self.lock)
        return rows

    def test_freeze_tracks_portably_and_refuses_changes_or_overwrites(self):
        rows = self._freeze()
        original_lock = self.lock.read_bytes()
        # Reordering rows and unavailable development assets cannot change evaluation.
        self._write([{"split": "train", "audio": "missing.wav"}, *reversed(rows)])
        tracks = verify_fulltrack_test(self.manifest, self.lock)
        self.assertEqual(20, len(tracks))
        self.assertEqual("oh a oh", tracks[0].text)
        self.assertEqual(TimedReferenceWord("oh", 0.7, 0.9), tracks[0].words[-1])
        self.assertEqual(((0.0, 0.2),), tracks[0].instrumental_intervals)
        self._write(rows)
        with self.assertRaises(FileExistsError):
            freeze_fulltrack_test(self.manifest, self.lock)
        self.assertEqual(original_lock, self.lock.read_bytes())

        audio = self.root / rows[0]["audio"]
        original_audio = audio.read_bytes()
        audio.write_bytes(original_audio[:-1] + b"\xff")
        with self.assertRaisesRegex(ValueError, "differ from the frozen test"):
            verify_fulltrack_test(self.manifest, self.lock)
        audio.write_bytes(original_audio)
        reference = self.root / rows[0]["reference"]
        reference.write_bytes(reference.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differ from the frozen test"):
            verify_fulltrack_test(self.manifest, self.lock)

    def test_development_checks_frozen_groups_without_opening_test_assets(self):
        test = self._freeze()[0]
        train = self._track(20, "train")
        validation = self._track(21, "validation")
        hidden = {"split": "test", "audio": "missing.wav", "reference": "missing.json"}
        self._write([train, validation, hidden])
        self.assertEqual(
            ["train", "validation"],
            [track.split for track in read_fulltrack_development(self.manifest, self.lock)],
        )
        for field, value, message in (
            ("artist_ids", ["artist-20", "ARTIST-0"], "artist_ids"),
            ("song_id", test["song_id"], "song_id"),
            ("lyric_id", test["lyric_id"], "lyric_id"),
            ("recording_family_id", test["recording_family_id"], "recording_family_id"),
            ("audio", test["audio"], "audio_sha256"),
            ("reference", test["reference"], "lyrics_sha256"),
        ):
            with self.subTest(field=field):
                self._write([dict(train, **{field: value}), validation, hidden])
                with self.assertRaisesRegex(ValueError, f"{message} overlaps the test set"):
                    read_fulltrack_development(self.manifest, self.lock)
        self._write([train, dict(validation, artist_ids=train["artist_ids"])])
        with self.assertRaisesRegex(ValueError, "artist_ids overlaps the train set"):
            read_fulltrack_development(self.manifest, self.lock)

    def test_train_only_untimed_vocal_clip(self):
        self._freeze()
        clip = dict(self._track(20, "train"), full_track=False, condition="vocal")
        validation = self._track(21, "validation")
        reference = self.root / clip["reference"]
        content = json.loads(reference.read_bytes())
        del content["words"]
        del content["instrumental_intervals"]
        reference.write_text(json.dumps(content), encoding="utf-8")
        self._write([clip, validation])
        track, held_out = read_fulltrack_development(self.manifest, self.lock)
        self.assertEqual((False, "vocal", "oh u oh", ()), (track.full_track, track.condition, track.text, track.words))
        self.assertEqual((True, "original_mix"), (held_out.full_track, held_out.condition))

        for split, condition in (("validation", "vocal"), ("test", "original_mix")):
            with self.subTest(split=split, condition=condition):
                self._write([dict(clip, split=split, condition=condition)])
                with self.assertRaisesRegex(ValueError, "complete English-language original mix"):
                    read_fulltrack_manifest(self.manifest)

    def test_validation_accepts_explicit_original_mix_excerpts(self):
        self._freeze()
        train = self._track(20, "train")
        validation = dict(self._track(21, "validation"), full_track=False)
        self._write([train, validation])
        track = read_fulltrack_development(self.manifest, self.lock)[1]
        self.assertEqual((track.split, track.condition, track.full_track), ("validation", "original_mix", False))

    def test_rewrapped_wav_is_still_a_duplicate_recording(self):
        test = self._freeze()[0]
        audio = (self.root / test["audio"]).read_bytes()
        metadata = b"JUNK\x04\x00\x00\x00note"
        length = int.from_bytes(audio[4:8], "little") + len(metadata)
        rewrapped = audio[:4] + length.to_bytes(4, "little") + audio[8:12] + metadata + audio[12:]
        (self.root / "rewrapped.wav").write_bytes(rewrapped)

        train, validation = self._track(20, "train"), self._track(21, "validation")
        self._write([dict(train, audio="rewrapped.wav"), validation])
        with self.assertRaisesRegex(ValueError, "pcm_sha256 overlaps the test set"):
            read_fulltrack_development(self.manifest, self.lock)

    def test_rejects_partial_audio_or_uncorroborated_and_inconsistent_reference(self):
        row = self._track(0)
        self._write([dict(row, source_frames=16000)])
        with self.assertRaisesRegex(ValueError, "source_frames"):
            read_fulltrack_manifest(self.manifest)
        self._write([dict(row, lyrics_permission="")])
        with self.assertRaisesRegex(ValueError, "lyrics_permission"):
            read_fulltrack_manifest(self.manifest)
        self._write([row])
        path = self.root / row["reference"]
        original = json.loads(path.read_bytes())
        for change, message in (
            ({"text": "oh a"}, "complete reference text"),
            ({"checked_by": "TRANSCRIBER"}, "separate reviewer"),
            ({"instrumental_intervals": [[0.1, 0.3]]}, "overlaps a lead-vocal word"),
        ):
            with self.subTest(change=change):
                path.write_text(json.dumps(dict(original, **change)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_fulltrack_manifest(self.manifest)

    def test_freeze_enforces_minimum_and_genres(self):
        rows = [self._track(index) for index in range(20)]
        diagnostic = [self._diagnostic(20, genre="rock"), self._diagnostic(21, genre="hip-hop"), self._diagnostic(22, genre="folk")]
        self._write([*rows[:-1], *diagnostic])
        with self.assertRaisesRegex(ValueError, "at least 20"):
            freeze_fulltrack_test(self.manifest, self.lock)
        self._write([*(dict(row, genre="pop") for row in rows), *diagnostic])
        with self.assertRaisesRegex(ValueError, "pop, rock, hip-hop"):
            freeze_fulltrack_test(self.manifest, self.lock)
        self.assertFalse(self.lock.exists())

    def test_freeze_requires_distinct_works_but_allows_untimed_references(self):
        rows = [self._track(index) for index in range(20)]
        self._write([*rows[:-1], dict(rows[-1], song_id=rows[0]["song_id"])])
        with self.assertRaisesRegex(ValueError, "song_id overlaps the test set"):
            freeze_fulltrack_test(self.manifest, self.lock)
        self._write(rows)
        reference = self.root / rows[0]["reference"]
        content = json.loads(reference.read_bytes())
        del content["words"]
        del content["instrumental_intervals"]
        reference.write_text(json.dumps(content), encoding="utf-8")
        freeze_fulltrack_test(self.manifest, self.lock)
        tracks = verify_fulltrack_test(self.manifest, self.lock)
        self.assertEqual(tracks[0].text, "oh a oh")
        self.assertEqual(tracks[0].words, ())
        self.assertIsNone(tracks[0].instrumental_intervals)

    def test_freeze_requires_a_positive_instrumental_annotation(self):
        rows = [self._track(index) for index in range(20)]
        for index, row in enumerate(rows):
            path = self.root / row["reference"]
            content = json.loads(path.read_bytes())
            if index % 2:
                del content["instrumental_intervals"]
            else:
                content["instrumental_intervals"] = []
            path.write_text(json.dumps(content), encoding="utf-8")
        self._write(rows)
        with self.assertRaisesRegex(ValueError, "annotated instrumental passages"):
            freeze_fulltrack_test(self.manifest, self.lock)
        self.assertFalse(self.lock.exists())

    def test_diagnostic_bank_freezes_without_lyrics_and_blocks_development_overlap(self):
        lexical = [self._track(index) for index in range(20)]
        for row in lexical:
            path = self.root / row["reference"]
            content = json.loads(path.read_bytes())
            del content["instrumental_intervals"]
            path.write_text(json.dumps(content), encoding="utf-8")
        auxiliary = [self._diagnostic(20), self._diagnostic(21)]
        self._write([*lexical, *auxiliary])
        freeze_fulltrack_test(self.manifest, self.lock)
        tracks = verify_fulltrack_test(self.manifest, self.lock)
        self.assertEqual(sum(not track.diagnostic_only for track in tracks), 20)
        checks = [track for track in tracks if track.diagnostic_only]
        self.assertEqual(len(checks), 2)
        for check in checks:
            self.assertEqual((check.text, check.words, check.instrumental_intervals), ("", (), ((0.15, 0.65),)))
        signatures = json.loads(self.lock.read_bytes())["tracks"]
        self.assertTrue(all("lyrics_sha256" not in item for item in signatures if item.get("diagnostic_only")))

        train = self._track(22, "train")
        validation = self._track(23, "validation")
        development = self.root / "development.jsonl"
        development.write_text("".join(json.dumps(row) + "\n" for row in [train, validation]), encoding="utf-8")
        self.assertEqual(len(read_fulltrack_development(development, self.lock)), 2)
        train["artist_ids"] = auxiliary[0]["artist_ids"]
        development.write_text("".join(json.dumps(row) + "\n" for row in [train, validation]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artist_ids overlaps the test set"):
            read_fulltrack_development(development, self.lock)

        annotation = self.root / auxiliary[0]["reference"]
        annotation.write_bytes(annotation.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differ from the frozen test"):
            verify_fulltrack_test(self.manifest, self.lock)

    def test_diagnostic_requires_complete_test_audio_and_checked_instrumentals(self):
        row = self._diagnostic(0)
        self._write([dict(row, full_track=False)])
        with self.assertRaisesRegex(ValueError, "complete English-language original mix"):
            read_fulltrack_manifest(self.manifest)
        self._write([dict(row, split="train")])
        with self.assertRaisesRegex(ValueError, "only be true for test"):
            read_fulltrack_manifest(self.manifest)

        self._write([row])
        path = self.root / row["reference"]
        reference = json.loads(path.read_bytes())
        for change, message in (
            ({"instrumental_intervals": []}, "require checked instrumental passages"),
            ({"text": "placeholder"}, "cannot contain lyric text"),
            ({"checked_by": ""}, "checked_by"),
        ):
            with self.subTest(change=change):
                path.write_text(json.dumps(dict(reference, **change)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_fulltrack_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
