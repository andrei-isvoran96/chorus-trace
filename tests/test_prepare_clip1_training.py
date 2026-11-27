import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import prepare_clip1_training as clip1
from chorustrace.fulltrack_data import read_fulltrack_manifest


class CLIP1StagingTests(unittest.TestCase):
    signal = "a" * 24
    pcm = b"\x00\x00\x01\x00" * 4000

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audit = self.root / "audit"
        self.audit.mkdir()
        self.archive = self.root / "audio.tar.gz"
        self.output = self.root / "output"

    @staticmethod
    def _add_member(archive, name, payload):
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    @staticmethod
    def _checksum(path):
        return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()

    def _make_inputs(self, *, license_label="Attribution", link=False, split="train"):
        train = split == "train"
        shortlist = self.audit / ("shortlist.jsonl" if train else "validation-shortlist.jsonl")
        shortlist.write_text(json.dumps({
            "signal": self.signal, "fma": 42, "artist.id": "7", "artist.name": "Fixture Artist",
            "track.title": "Fixture Track", "track.genre_top": "Pop",
            "clip1_license": license_label, "track.license": license_label,
        }) + "\n", encoding="utf-8")
        license_file = self.audit / clip1.LICENSE_NAME
        license_file.write_text(f"fma_track_id,licence\n42,{license_label}\n", encoding="utf-8")
        metadata_file = self.audit / clip1.METADATA_NAME
        metadata = json.dumps([{
            "signal": self.signal, "fma": "000042", "original_prompt": "Oh, hello moon!", "prompt": "ignored label",
        }]).encode()
        with tarfile.open(metadata_file, "w:gz") as archive:
            member = clip1.METADATA_MEMBER if train else clip1.VALIDATION_METADATA_MEMBER
            self._add_member(archive, member, metadata)
        packed = (8000 << 44) | (15 << 36) | 8000
        flac = b"fLaC\x80\x00\x00\x22" + bytes(10) + packed.to_bytes(8, "big") + clip1._md5(self.pcm).digest()
        source_split = "train" if train else "valid"
        source_member = f"cadenza_data/{source_split}/unprocessed/{self.signal}_unproc.flac"
        with tarfile.open(self.archive, "w:gz") as archive:
            self._add_member(archive, f"cadenza_data/{source_split}/signals/{self.signal}.flac", b"processed decoy")
            if link:
                item = tarfile.TarInfo(source_member)
                item.type = tarfile.SYMTYPE
                item.linkname = "untrusted-local-path"
                archive.addfile(item)
            else:
                self._add_member(archive, source_member, flac)
        checksums = {
            ("training_archive" if train else "validation_archive"): self._checksum(self.archive),
            "metadata_archive": self._checksum(metadata_file),
            "license_catalog": self._checksum(license_file),
        }
        return checksums, clip1._md5(metadata).hexdigest(), flac

    def test_stages_only_original_unprocessed_train_audio_with_evidence(self):
        checksums, metadata_md5, flac = self._make_inputs()
        with (
            patch.object(clip1, "RELEASE_MD5S", checksums),
            patch.object(clip1, "METADATA_MEMBER_MD5", metadata_md5),
            patch.object(clip1.subprocess, "run", return_value=SimpleNamespace(stdout=self.pcm)) as ffmpeg,
        ):
            manifest = clip1.prepare(self.audit, self.archive, self.output)
        self.assertEqual(ffmpeg.call_args.kwargs["input"], flac)
        row = json.loads(manifest.read_text(encoding="utf-8"))
        track, = read_fulltrack_manifest(manifest)
        self.assertEqual(track.text, "oh hello moon")
        self.assertEqual(track.artist_ids, ("fixture artist", "fma:artist:7"))
        self.assertEqual((track.song_id, track.lyric_id, track.recording_family_id), ("fma:42",) * 3)
        self.assertEqual((track.split, track.condition, track.full_track), ("train", "original_mix", False))
        self.assertIn("Attribution", row["lyrics_permission"])
        self.assertEqual(row["source_flac_sha256"], hashlib.sha256(flac).hexdigest())
        with wave.open(str(track.audio), "rb") as wav:
            self.assertEqual(wav.readframes(8000), self.pcm)
        report = json.loads((self.output / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["duration_seconds"], 1.0)
        self.assertTrue(report["verified_gzip_crc"])
        self.assertEqual(report["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_stages_validation_member_and_manifest_without_train_inputs(self):
        checksums, metadata_md5, flac = self._make_inputs(split="validation")
        with (
            patch.object(clip1, "RELEASE_MD5S", checksums),
            patch.object(clip1, "VALIDATION_METADATA_MEMBER_MD5", metadata_md5),
            patch.object(clip1.subprocess, "run", return_value=SimpleNamespace(stdout=self.pcm)) as ffmpeg,
        ):
            manifest = clip1.prepare(self.audit, self.archive, self.output, split="validation")
        self.assertEqual(ffmpeg.call_args.kwargs["input"], flac)
        self.assertEqual(manifest.name, "validation.jsonl")
        track, = read_fulltrack_manifest(manifest)
        self.assertEqual((track.split, track.condition, track.full_track), ("validation", "original_mix", False))
        row = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertIn("cadenza_clip1_data.valid.v1.2.tar.gz", row["audio_source"])
        self.assertIn("/valid/unprocessed/", row["audio_source"])
        report = json.loads((self.output / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual((report["split"], report["release_md5"]), ("validation", checksums))

    def test_rejects_unsafe_member_and_does_not_publish_partial_dataset(self):
        checksums, metadata_md5, _ = self._make_inputs(link=True)
        with (
            patch.object(clip1, "RELEASE_MD5S", checksums),
            patch.object(clip1, "METADATA_MEMBER_MD5", metadata_md5),
            patch.object(clip1.subprocess, "run") as ffmpeg,
        ):
            with self.assertRaisesRegex(ValueError, "invalid or repeated selected FLAC archive member"):
                clip1.prepare(self.audit, self.archive, self.output)
        ffmpeg.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_rejects_disallowed_license_and_pcm_checksum_mismatch(self):
        checksums, metadata_md5, flac = self._make_inputs(license_label="Attribution-NoDerivatives")
        with patch.object(clip1, "RELEASE_MD5S", checksums), patch.object(clip1, "METADATA_MEMBER_MD5", metadata_md5):
            with self.assertRaisesRegex(ValueError, "approved source term"):
                clip1.prepare(self.audit, self.archive, self.output)
        with patch.object(clip1.subprocess, "run", return_value=SimpleNamespace(stdout=bytes(len(self.pcm)))):
            with self.assertRaisesRegex(ValueError, "STREAMINFO checksum"):
                clip1._decode_flac(flac, self.output / "fixture.wav", "ffmpeg", max_seconds=12)
        self.assertFalse(self.output.exists())

    def test_requires_opt_in_for_noncommercial_sharealike_and_records_terms(self):
        license_label = "Attribution-Noncommercial-Share Alike 2.0 UK: England "
        checksums, metadata_md5, _ = self._make_inputs(license_label=license_label)
        with (
            patch.object(clip1, "RELEASE_MD5S", checksums),
            patch.object(clip1, "METADATA_MEMBER_MD5", metadata_md5),
            patch.object(clip1.subprocess, "run", return_value=SimpleNamespace(stdout=self.pcm)) as ffmpeg,
        ):
            with self.assertRaisesRegex(ValueError, "approved source term"):
                clip1.prepare(self.audit, self.archive, self.output)
            ffmpeg.assert_not_called()
            manifest = clip1.prepare(self.audit, self.archive, self.output, allow_noncommercial=True)
        row = json.loads(manifest.read_text(encoding="utf-8"))
        for field in ("recording_permission", "lyrics_permission"):
            self.assertIn("Noncommercial research only", row[field])
            self.assertIn(license_label.strip(), row[field])
        report = json.loads((self.output / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report["noncommercial_source_clips"], 1)

    def test_exact_catalog_categories_only(self):
        for label, expected in (
            ("Attribution-NonCommercial", True),
            ("Attribution-NonCommercial 2.5", True),
            ("Attribution-NonCommercial-ShareAlike 3.0 (Will Not Appear in Vimeo Music Store)", True),
            ("Attribution-Noncommercial-Share Alike 3.0 United States", True),
            ("Attribution-NonCommercial-NoDerivs 3.0 Italy", False),
            ("Attribution-NoDerivs 2.5 Canada", False),
            ("Public Domain / Sound Recording Common Law Protection", False),
            ("Public Domain Mark 1.0", False),
            ("Public Domain", False),
            ("Attribution-NonCommercial-ShareAlike with added restrictions", False),
            (None, False),
        ):
            with self.subTest(label=label):
                self.assertFalse(clip1.is_allowed_license(label))
                self.assertEqual(clip1.is_allowed_license(label, allow_noncommercial=True), expected)
        self.assertTrue(clip1.is_allowed_license("CC0 1.0 Universal"))


if __name__ == "__main__":
    unittest.main()
