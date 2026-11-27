import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from benchmarks.acquire_frozen_fulltrack import _audio_index, _stage, acquire


class FrozenAcquisitionTests(unittest.TestCase):
    def test_pinned_source_checks_content_and_length(self):
        payload = b"original release"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            bundled = output / "bundled"
            bundled.write_bytes(payload)
            _stage("unused", output / "good", bundled, digest, size=len(payload))
            self.assertEqual((output / "good").read_bytes(), payload)
            for number, sha, size in ((0, "0" * 64, len(payload)), (1, digest, 0)):
                with self.assertRaisesRegex(ValueError, "pinned source checksum or size differs"):
                    _stage("unused", output / str(number), bundled, sha, size=size)

    def test_lfs_index_verifies_digest_independent_of_row_order_and_directory_names(self):
        entries = [(f"track-{index:02d}.mp3", f"{index:064x}", index + 10) for index in range(20)]
        digest = hashlib.sha256(json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        rows = [{"path": f"subsets/en/audio/{name}", "lfs": {"oid": sha}, "size": size}
                for name, sha, size in reversed(entries)]
        with (patch("benchmarks.acquire_frozen_fulltrack._MP3_INDEX_SHA", digest),
              patch("benchmarks.acquire_frozen_fulltrack._fetch", return_value=BytesIO(json.dumps(rows).encode()))):
            self.assertEqual(_audio_index("unused", None), {name: (sha, size) for name, sha, size in entries})
        rows[0]["size"] += 1
        with (patch("benchmarks.acquire_frozen_fulltrack._MP3_INDEX_SHA", digest),
              patch("benchmarks.acquire_frozen_fulltrack._fetch", return_value=BytesIO(json.dumps(rows).encode()))):
            with self.assertRaisesRegex(ValueError, "MP3 index differs"):
                _audio_index("unused", None)

    def test_refuses_to_overwrite_existing_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary)
            old = tree / "marker"
            old.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                acquire(tree)
            self.assertEqual(old.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
