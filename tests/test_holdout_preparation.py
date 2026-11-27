import unittest

from benchmarks.prepare_jam_alt_holdout import _lead_text
from benchmarks.screen_clip1_development import _screen


class HoldoutPreparationTests(unittest.TestCase):
    def test_reviewed_lead_preserves_repeats_and_corroborates_number(self):
        reviewed = "Follow me (background)\nFollow me\nWhen you are 29 walk to town"
        original = [{"text": word} for word in "when you are twenty nine walk to town".split()]
        self.assertEqual(_lead_text(reviewed, original),
                         "follow me follow me when you are twenty nine walk to town")
        with self.assertRaisesRegex(ValueError, "corroborate"):
            _lead_text(reviewed, original[:-1])

    def test_screening_excludes_held_out_artist_work_and_lyric_text(self):
        rows = [{"signal": str(index), "fma": str(index), "hearing_loss": "No Loss",
                 "original_prompt": prompt}
                for index, prompt in enumerate(("another green path", "through the long night",
                                                "across the empty road", "words from the holdout"))]
        catalog = {index: {"artist.name": artist, "track.title": title, "track.license": "Attribution"}
                   for index, (artist, title) in enumerate((("New Singer", "New Track"),
                                                            ("Held Singer", "Other Work"),
                                                            ("Other Singer", "Held Work (Live)"),
                                                            ("Third Singer", "Third Work")))}
        selected = _screen(rows, catalog, {index: "Attribution" for index in catalog},
                           {"heldsinger"}, {"heldwork"}, [" these are words from the holdout today "])
        self.assertEqual([row["signal"] for row in selected], ["0"])

    def test_screening_excludes_individual_collaborators(self):
        rows = [{"signal": "signal", "fma": "1", "hearing_loss": "No Loss", "original_prompt": "a quiet morning"}]
        catalog = {1: {"artist.name": "Other Artist & Held Singer", "track.title": "New Work",
                       "track.license": "Attribution"}}
        self.assertEqual(_screen(rows, catalog, {1: "Attribution"}, {"heldsinger"}, set(), []), [])
