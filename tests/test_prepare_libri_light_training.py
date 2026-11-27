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

from benchmarks import prepare_libri_light_training as libri
from chorustrace.fulltrack_data import read_fulltrack_manifest


class LibriLightStagingTests(unittest.TestCase):
    @staticmethod
    def _member(archive, name, content):
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    def test_stages_only_one_hour_disjoint_audio_and_reports_ctc_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "download.tgz"
            folder = "librispeech_finetuning/1h/0/clean/23/45"
            transcript = (
                "23-45-0001 FIRST TEN LETTERS ARE ABCDEFGHIJ ABCDEFGHIJ ABCDEFGHIJ ABCDEFGHIJ ABCDEFGHIJ\n"
                "23-45-0002 ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT CAME LAST\n"
            )
            with tarfile.open(archive, "w:gz") as contents:
                self._member(contents, "librispeech_finetuning/9h/clean/1/1/1-1-1.flac", b"ignored")
                self._member(contents, f"{folder}/23-45.trans.txt", transcript.encode())
                self._member(contents, f"{folder}/23-45-0001.flac", b"source kept")
                self._member(contents, f"{folder}/23-45-0002.flac", b"source protected")
            validation, lock = root / "validation.jsonl", root / "test.lock.json"
            validation.write_bytes(b"validation")
            lock.write_bytes(b"test lock")
            protected = SimpleNamespace(
                track_id="val", song_id="valwork", lyric_id="valwords", recording_family_id="valrecording",
                artist_ids=("validation artist",), split="validation",
                text="tomorrow alpha bravo charlie delta echo foxtrot come first",
            )

            def validation_manifest(path):
                return [protected] if path == validation else read_fulltrack_manifest(path)

            def decode(source, path, _ffmpeg, _max_seconds):
                self.assertEqual(source, b"source kept")
                pcm = b"\x01\x00" * 32000
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 16000, 32000, "NONE", "not compressed"))
                    wav.writeframes(pcm)
                return hashlib.md5(pcm, usedforsecurity=False).hexdigest()

            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            with (
                patch.object(libri, "ARCHIVE_SHA256", checksum),
                patch.object(libri, "read_fulltrack_manifest", side_effect=validation_manifest),
                patch.object(libri, "verify_fulltrack_test", return_value=[]),
                patch.object(libri, "_decode_flac", side_effect=decode) as ffmpeg,
            ):
                manifest = libri.prepare(archive, validation, root / "test.jsonl", lock, root / "staged")
            ffmpeg.assert_called_once()
            row = json.loads(manifest.read_bytes())
            track, = read_fulltrack_manifest(manifest)
            self.assertEqual(track.track_id, "librispeech:23-45-0001")
            self.assertEqual(track.song_id, "librispeech:chapter:45")
            self.assertEqual(track.artist_ids, ("librispeech:speaker:23",))
            self.assertEqual((track.split, track.condition, track.full_track), ("train", "vocal", False))
            self.assertEqual(json.loads((manifest.parent / row["reference"]).read_bytes())["text"],
                             transcript.splitlines()[0].partition(" ")[2])
            report = json.loads((manifest.parent / "verification.json").read_bytes())
            self.assertEqual(report["source_utterances"], 2)
            self.assertEqual(report["held_out_overlap_exclusions"], 1)
            self.assertEqual(report["ctc_fit_counts"], {"320": 1, "480": 0})

    def test_rejects_link_masquerading_as_one_hour_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "download.tgz"
            with tarfile.open(archive, "w:gz") as contents:
                link = tarfile.TarInfo("librispeech_finetuning/1h/0/clean/23/45/23-45-0001.flac")
                link.type = tarfile.SYMTYPE
                link.linkname = "/unrelated.flac"
                contents.addfile(link)
            with self.assertRaisesRegex(ValueError, "invalid one-hour subset member"):
                libri._read_sources(archive, root)


if __name__ == "__main__":
    unittest.main()
