import unittest
from pathlib import Path
from unittest.mock import Mock

from chorustrace.fulltrack_data import FullTrack, TimedReferenceWord
from chorustrace.fulltrack_decode import FullTrackTranscript, TimedWord
from chorustrace.fulltrack_evaluation import evaluate_full_tracks


class NativeVerificationTests(unittest.TestCase):
    def test_word_emitted_during_instrumental_cannot_match_repeated_lyric(self):
        track = FullTrack(
            track_id="fixture", song_id="song", artist_ids=("artist",), lyric_id="lyric",
            recording_family_id="family", genre="rock", split="test", audio=Path("mix.wav"),
            text="hold hold", duration_seconds=5.0,
            words=(TimedReferenceWord("hold", 0.2, 0.5), TimedReferenceWord("hold", 4.1, 4.4)),
            instrumental_intervals=((1.0, 4.0),),
        )
        recognizer = Mock()
        first_word = TimedWord("hold", 0.2, 0.5)
        recognizer.transcribe_full_track.side_effect = [
            FullTrackTranscript("hold hold", (first_word, TimedWord("hold", 2.0, 2.3)), ()),
            FullTrackTranscript("hold hold", (first_word, TimedWord("hold", 4.1, 4.4)), ()),
        ]

        report = evaluate_full_tracks(recognizer, [track], compare_unsegmented=True)

        for scores in (report["tracks"][0], report["aggregate"]):
            self.assertEqual(scores["reference_words"], 2)
            self.assertEqual(scores["substitutions"], 0)
            self.assertEqual(scores["deletions"], 1)
            self.assertEqual(scores["insertions"], 1)
            self.assertEqual(scores["word_edits"], 2)
            self.assertEqual(scores["wer"], 1.0)
            self.assertEqual(scores["instrumental"]["words"], 1)
        paired = report["aggregate"]["segment_boundaries"]["paired"]
        self.assertEqual(paired["unsegmented_word_edits"], 0)
        self.assertEqual(paired["segmentation_only_word_edits"], 2)
