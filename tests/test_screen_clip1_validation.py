import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import screen_clip1_validation as screen
from benchmarks.prepare_clip1_training import LICENSE_NAME, METADATA_NAME, METADATA_MEMBER, VALIDATION_METADATA_MEMBER
from chorustrace.data import read_jsonl, write_jsonl


class CLIP1ValidationSelectionTests(unittest.TestCase):
    def test_reserves_training_and_test_while_selecting_licensed_hip_hop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, current_training, output = root / "source", root / "training", root / "new"
            audit.mkdir()
            current_training.mkdir()
            fixed = current_training / "shortlist.jsonl"
            write_jsonl(fixed, [{"signal": "t", "fma": 1, "artist.id": "100", "artist.name": "Train & Guest",
                                 "track.title": "Canonical Work"}])
            original = fixed.read_bytes()
            with (audit / LICENSE_NAME).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["fma_track_id", "licence"])
                writer.writerows((index, "Attribution-NonCommercial") for index in range(1, 9))

            def row(signal, fma, text):
                return {"signal": signal, "fma": str(fma), "original_prompt": text, "hearing_loss": "No Loss"}

            valid = [
                row("overlap-work", 1, "Different line from the same work"),
                row("overlap-artist", 2, "Matching training performer alias"),
                row("overlap-catalog-id", 3, "Alias unknown but catalog artist matches"),
                row("overlap-text", 4, "Please greet the winter moon tonight"),
                row("overlap-test-text", 5, "I only sleep under city lights"),
                row("overlap-test-artist", 6, "A different performer appears below"),
                row("eligible", 7, "I paint new patterns daily"),
                row("overlap-title", 8, "A claimed title matches after normalization"),
            ]
            with tarfile.open(audit / METADATA_NAME, "w:gz") as archive:
                for member, rows in ((METADATA_MEMBER, [row("t", 1, "Greet the winter moon")]),
                                     (VALIDATION_METADATA_MEMBER, valid)):
                    payload = json.dumps(rows).encode()
                    info = tarfile.TarInfo(member)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            catalog = {
                index: {"artist.id": str(index), "artist.name": f"Performer {chr(97 + index)}",
                        "track.title": f"Work {chr(97 + index)}", "track.genre_top": "Hip-Hop",
                        "track.license": "Attribution-NonCommercial"}
                for index in range(1, 9)
            }
            catalog[2]["artist.name"] = "Guest"
            catalog[3]["artist.id"] = "100"
            catalog[6]["artist.name"] = "Test Artist"
            catalog[8]["track.title"] = "Canonical Work (acoustic)"
            holdout = [SimpleNamespace(artist_ids=("test artist",), song_id="test:moonlight", diagnostic_only=False,
                                      text="now i only sleep under city lights when it is late")]
            with patch.object(screen, "verify_fulltrack_test", return_value=holdout), patch.object(
                screen, "_catalog", return_value=catalog,
            ):
                summary = screen.prepare(audit, current_training, root / "test.jsonl", root / "lock.json", output)
            self.assertEqual(summary["validation_genres"], {"Hip-Hop": 1})
            self.assertEqual([row["signal"] for _, row in read_jsonl(output / "validation-shortlist.jsonl")], ["eligible"])
            self.assertEqual(fixed.read_bytes(), original)
            self.assertEqual((output / "shortlist.jsonl").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
