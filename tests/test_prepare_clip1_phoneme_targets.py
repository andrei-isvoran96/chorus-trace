import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from benchmarks import prepare_clip1_phoneme_targets as targets
from chorustrace.data import write_jsonl


class CLIP1PhonemeTargetTests(unittest.TestCase):
    def test_prepares_case_preserving_targets_without_opening_heldout_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            signal, valid_signal = "a" * 24, "c" * 24
            wav = root / "clip.wav"
            with wave.open(str(wav), "wb") as stream:
                stream.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
                stream.writeframes(b"\x00\x00" * 16000)
            (root / "ref.json").write_text(json.dumps({
                "text": "Hello, Moon!", "complete_lead_vocal": True, "transcribed_by": "fixture",
            }), encoding="utf-8")
            manifest = root / "dev.jsonl"
            train = {
                "track_id": f"clip1:{signal}", "song_id": "fma:17", "lyric_id": "fma:17",
                "recording_family_id": "fma:17", "artist_ids": ["train performer"], "split": "train",
                "genre": "rock", "condition": "original_mix", "language": "en", "full_track": False,
                "audio": "clip.wav", "reference": "ref.json", "source_frames": 16000, "source_sample_rate": 16000,
                "audio_source": f"clip1 fixture :: cadenza_data/train/unprocessed/{signal}_unproc.flac",
                "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "recording_permission": "fixture audio", "lyrics_permission": "fixture lyrics",
            }
            validation = dict(train, track_id=f"clip1:{valid_signal}", song_id="fma:18", lyric_id="fma:18",
                              recording_family_id="fma:18", artist_ids=["validation performer"], split="validation",
                              audio="missing-validation.wav", reference="missing-validation.json")
            write_jsonl(manifest, [train, validation, {"split": "test", "audio": "missing-test.wav"}])
            lock = root / "test.lock.json"
            lock.write_bytes(b"private lock fixture")
            archive = root / "metadata.tar.gz"
            label = {
                "signal": signal, "fma": "000017", "hearing_loss": "No Loss",
                "original_prompt": "Hello, Moon!", "prompt": "hello moon",
                "prompt_pronunciation": "HH-HH ah", "n_words": 2, "n_phonemes": 3,
            }
            metadata = json.dumps([label, dict(label, signal=valid_signal)]).encode()
            with tarfile.open(archive, "w:gz") as package:
                member = tarfile.TarInfo(targets.METADATA_MEMBER)
                member.size = len(metadata)
                package.addfile(member, io.BytesIO(metadata))
            with (
                patch.object(targets, "_read_lock", return_value=[]),
                patch.object(targets, "_verify_file", return_value="fixture release checksum"),
                patch.object(targets, "METADATA_MEMBER_MD5", targets._md5(metadata).hexdigest()),
            ):
                sidecar = targets.prepare(manifest, lock, archive, root / "staged")
                label_mismatch = dict(train, audio_sha256="wrong")
                write_jsonl(manifest, [label_mismatch, validation])
                with self.assertRaisesRegex(ValueError, "WAV differs"):
                    targets.prepare(manifest, lock, archive, root / "rejected")
            payload = json.loads(sidecar.read_bytes())
            self.assertEqual(payload["vocabulary"], ["HH", "ah"])
            self.assertEqual(payload["examples"], {f"clip1:{signal}": {
                "phones": [1, 1, 2], "grapheme_sha256": hashlib.sha256(b"hello moon").hexdigest(),
            }})
            report = json.loads((sidecar.parent / "verification.json").read_bytes())
            self.assertEqual(report["ctc"]["320"]["minimum_spare_frames"], 46)
            self.assertEqual(report["case_variant_examples"], 1)
            self.assertFalse((root / "rejected").exists())

    def test_rejects_malformed_groups_and_inline_alternative_labels(self):
        for string in ("hh--ah m", "hh/ah m", "hh-ah  m", "hh-ah-m", "<unknown> m"):
            with self.subTest(pronunciation=string):
                with self.assertRaisesRegex(ValueError, "malformed phoneme groups"):
                    targets._phonemes({"prompt_pronunciation": string, "prompt": "a b",
                                      "n_words": 2, "n_phonemes": 3}, "fixture")


if __name__ == "__main__":
    unittest.main()
