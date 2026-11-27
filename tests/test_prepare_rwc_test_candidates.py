import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from benchmarks import prepare_rwc_test_candidates as rwc


class _Response(io.BytesIO):
    def __init__(self, body, content_range, status=206):
        super().__init__(body)
        self.headers = {"Content-Range": content_range}
        self.status = status


class RWCCandidateTests(unittest.TestCase):
    def test_stages_full_member_and_reuses_crc_verified_wav(self):
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wav:
            wav.setparams((1, 1, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(range(100)) * 80)
        original = wav_bytes.getvalue()
        zip_bytes = io.BytesIO()
        with ZipFile(zip_bytes, "w", compression=ZIP_DEFLATED) as zip_file:
            zip_file.writestr("RWC-P/RWC_P001.wav", b"skip the unrelated member")
            zip_file.writestr("RWC-P/RWC_P002.wav", original)
        archive_bytes = zip_bytes.getvalue()
        with ZipFile(io.BytesIO(archive_bytes)) as zip_file:
            directory = zip_file.start_dir
            selected = zip_file.getinfo("RWC-P/RWC_P002.wav").header_offset
        request_spans = []

        def open_range(request, *, timeout):
            span = request.get_header("Range").removeprefix("bytes=")
            start, end = map(int, span.split("-"))
            request_spans.append((start, end + 1))
            return _Response(archive_bytes[start:end + 1], f"bytes {span}/{len(archive_bytes)}")

        row = {
            "CollID": "P", "RWCID": "RWC_P002", "Title": "Fixture", "Artist": "Singer",
            "GenreMain": "Popular", "GenreSub": "Pop", "duration": "1.0", "genre": "pop",
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(rwc, "ARCHIVE_SIZES", {"P": len(archive_bytes)}),
            patch.object(rwc, "ZIP_TAIL_SIZE", len(archive_bytes) - directory),
            patch.object(rwc, "select_candidates", return_value=[row]),
            patch.object(rwc, "urlopen", side_effect=open_range),
        ):
            root = Path(temporary)
            metadata = root / "metadata.csv"
            metadata.write_text("fixture", encoding="utf-8")
            inventory = rwc.prepare(metadata, root / "stage")
            payload = json.loads(inventory.read_text(encoding="utf-8"))
            self.assertEqual(payload["split"], "test")
            track = payload["tracks"][0]
            audio_path = inventory.parent / track["audio"]
            self.assertEqual(audio_path.read_bytes(), original)
            self.assertEqual(track["sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual((track["frames"], track["sample_rate"], track["duration_seconds"]), (8000, 8000, 1.0))
            self.assertEqual(request_spans, [(directory, len(archive_bytes)), (selected, directory)])

            request_spans.clear()
            rwc.prepare(metadata, root / "stage")
            self.assertEqual(request_spans, [(directory, len(archive_bytes))])
            audio_path.write_bytes(original[:-1] + b"\xff")
            with self.assertRaisesRegex(ValueError, "differ from the release ZIP entry"):
                rwc.prepare(metadata, root / "stage")

    def test_requires_complete_exact_partial_response(self):
        url = "https://example.invalid/archive.zip"
        for response, message in (
            (_Response(b"whole zip", "bytes 2-3/4", 200), "expected HTTP 206"),
            (_Response(b"ab", "bytes 0-1/4"), "expected HTTP 206"),
            (_Response(b"c", "bytes 2-3/4"), "truncated byte range"),
            (_Response(b"cde", "bytes 2-3/4"), "extra bytes"),
        ):
            with self.subTest(message=message), patch.object(rwc, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, message):
                    rwc._copy_range(url, 4, 2, 4, io.BytesIO())


if __name__ == "__main__":
    unittest.main()
