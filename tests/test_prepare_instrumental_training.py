import hashlib
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from benchmarks import prepare_instrumental_training as prepare
from benchmarks.prepare_rwc_instrumental_development import read_bank
from chorustrace.data import read_jsonl, write_jsonl
from chorustrace.fulltrack_data import _audio_signature, read_fulltrack_manifest


class PrepareInstrumentalTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.jsonl"
        self.development = self.root / "development.jsonl"
        self.lock = self.root / "lock.json"
        self.lock.write_text("fixture lock", encoding="utf-8")

    def _row(self, tag, artist, split, samples, *, diagnostics=False):
        audio, reference = self.root / f"{tag}.wav", self.root / f"{tag}.json"
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 1, 100, 1000, "NONE", "not compressed"))
            stream.writeframes(bytes([samples]) * 1000)
        metadata = {
            "track_id": tag, "song_id": f"song {tag}", "lyric_id": f"lyric {tag}",
            "recording_family_id": f"recording {tag}", "artist_ids": [artist], "genre": "rock", "split": split,
            "audio": audio.name, "reference": reference.name, "audio_source": "fixture recording",
            "recording_permission": "fixture noncommercial terms", "source_frames": 1000, "source_sample_rate": 100,
            "full_track": diagnostics, "condition": "original_mix",
        }
        if diagnostics:
            evidence = {
                "checked_by": "published annotator", "review_evidence": prepare.METHOD,
                "source_activity_license": prepare._ACTIVITY_LICENSE,
                "source_activity_file_sha256": "a" * 64, "source_activity_archive_sha256": "b" * 64,
                "instrumental_intervals": [[1.023, 9.872]],
            }
            metadata["diagnostic_only"] = True
        else:
            evidence = {"transcribed_by": "fixture", "text": f"sung phrase {tag}", "complete_lead_vocal": True}
            metadata.update(language="en", lyrics_permission="fixture lyrics")
        reference.write_text(json.dumps(evidence), encoding="utf-8")
        _, metadata["audio_sha256"], metadata["pcm_sha256"] = _audio_signature(audio, metadata, tag)
        metadata["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
        return metadata

    def test_reserves_alias_linked_groups_and_reuses_only_checked_wav_ranges(self):
        a = self._row("negative a", "Solo Artist", "validation", 1, diagnostics=True)
        b = self._row("negative b", "soloartist", "validation", 2, diagnostics=True)
        c = self._row("negative c", "Other Performer", "validation", 3, diagnostics=True)
        write_jsonl(self.source, [a, b, c])
        train = self._row("train item", "Music Train", "train", 4)
        validation = self._row("validation item", "Music Valid", "validation", 5)
        write_jsonl(self.development, [train, validation])
        with patch.object(prepare, "_read_lock", return_value=[]), patch(
            "chorustrace.fulltrack_data._read_lock", return_value=[],
        ):
            output = prepare.prepare(self.source, self.development, self.lock, self.root / "out", groups=1)
        tracks = read_fulltrack_manifest(output)
        active = {track.track_id for track in tracks if track.condition == "instrumental"}
        reserved = {track.track_id for track in read_bank(output.parent / "validation-diagnostics.jsonl")}
        self.assertTrue(active and reserved and not (active & reserved))
        self.assertIn({a["track_id"], b["track_id"]}, (active, reserved))
        self.assertEqual({track.track_id for track in tracks} - active, {train["track_id"], validation["track_id"]})
        self.assertEqual({row["track_id"] for _, row in read_jsonl(output.parent / "train-negatives.jsonl")}, active)
        for track in tracks:
            if track.track_id not in active:
                continue
            self.assertFalse(track.text)
            self.assertEqual(len(track.instrumental_intervals), 2)
            for start, end in track.instrumental_intervals:
                self.assertGreaterEqual(math.floor(start * 100), math.ceil(1.023 * 100))
                self.assertLessEqual(math.ceil(end * 100), math.floor(9.872 * 100))
                self.assertLessEqual(math.ceil(end * 100) - math.floor(start * 100), 600)

    def test_rejects_bank_annotation_without_published_evidence(self):
        row = self._row("bank item", "Artist", "validation", 1, diagnostics=True)
        path = self.root / row["reference"]
        data = json.loads(path.read_bytes())
        del data["source_activity_file_sha256"]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "published, human-annotated"):
            prepare._validate_annotation(self.source, row)


if __name__ == "__main__":
    unittest.main()
