import unittest
from pathlib import Path

from chorustrace.evaluation import edit_distance
from chorustrace.fulltrack_data import FullTrack, TimedReferenceWord
from chorustrace.fulltrack_decode import FullTrackTranscript, TimedWord
from chorustrace.fulltrack_evaluation import evaluate_full_tracks


class FakeRecognizer:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def transcribe_full_track(self, audio, *, segment_seconds):
        self.calls.append((audio, segment_seconds))
        return self.results[audio, segment_seconds]


def track(track_id, text, *, words=(), instrumental_intervals=None):
    return FullTrack(
        track_id=track_id, song_id=track_id, artist_ids=(track_id,), lyric_id=track_id,
        recording_family_id=track_id, genre="rock", split="test", audio=Path(f"{track_id}.wav"),
        text=text, duration_seconds=6.0, words=words, instrumental_intervals=instrumental_intervals,
    )


def transcript(text, starts, joins=()):
    words = tuple(TimedWord(word, start, start + 0.1) for word, start in zip(text.split(), starts, strict=True))
    return FullTrackTranscript(text, words, joins)


class FullTrackEvaluationTests(unittest.TestCase):
    def test_counts_edits_across_complete_transcripts_and_micro_averages(self):
        first = track("full-1", "One two three four five six.")
        second = track("full-2", "chorus chorus")
        prediction = "one too three five six again"
        recognizer = FakeRecognizer({
            (first.audio, 20.0): transcript(prediction, range(6)),
            (second.audio, 20.0): transcript("chorus chorus", (1.0, 4.0)),
        })

        report = evaluate_full_tracks(recognizer, [first, second], samples=1)
        first_score, second_score = report["tracks"]
        self.assertEqual(recognizer.calls, [(first.audio, 20.0), (second.audio, 20.0)])
        self.assertEqual(report["track_count"], 2)
        self.assertEqual(first_score["substitutions"], 1)
        self.assertEqual(first_score["deletions"], 1)
        self.assertEqual(first_score["insertions"], 1)
        self.assertEqual(first_score["wer"], 3 / 6)
        self.assertEqual(second_score["wer"], 0.0)
        self.assertEqual(report["aggregate"]["reference_words"], 8)
        self.assertEqual(report["aggregate"]["wer"], 3 / 8)
        reference = "one two three four five six"
        self.assertEqual(report["aggregate"]["cer"], edit_distance(reference, prediction) /
                         (len(reference) + len(second.text)))
        self.assertEqual(report["samples"], [{"track_id": first.track_id, "reference": reference,
                                              "prediction": prediction}])

    def test_counts_instrumental_words_and_compares_matched_boundary_edits(self):
        first = track("full-1", "sing sing here again", words=(
            TimedReferenceWord("sing", 0.1, 0.2), TimedReferenceWord("sing", 0.5, 0.6),
            TimedReferenceWord("here", 2.0, 2.1), TimedReferenceWord("again", 3.0, 3.1),
        ), instrumental_intervals=((1.0, 1.95), (4.0, 6.0)))
        unknown = track("full-2", "end")
        recognizer = FakeRecognizer({
            (first.audio, 2.0): transcript("sing sing filler wrong again", (0.1, 0.5, 1.75, 2, 3), (2, 4)),
            (first.audio, None): transcript("sing sing here oops", (0.1, 0.5, 2, 3)),
            (unknown.audio, 2.0): transcript("end", (5.0,), (2, 4)),
            (unknown.audio, None): transcript("end", (5.0,)),
        })

        report = evaluate_full_tracks(recognizer, [first, unknown], segment_seconds=2.0,
                                      compare_unsegmented=True, samples=0)
        aggregate = report["aggregate"]
        self.assertEqual(report["samples"], [])
        self.assertEqual(aggregate["instrumental_annotated_tracks"], 1)
        self.assertEqual(aggregate["instrumental_annotated_duration_seconds"], 6.0)
        self.assertEqual(aggregate["instrumental_tracks_with_intervals"], 1)
        self.assertEqual(aggregate["instrumental"]["words"], 1)
        self.assertEqual(aggregate["instrumental"]["interval_false_alarm_rate"], 0.5)
        self.assertAlmostEqual(aggregate["instrumental"]["words_per_minute"], 60 / 2.95)
        self.assertIsNone(report["tracks"][1]["instrumental"])
        boundaries = aggregate["segment_boundaries"]
        self.assertEqual(boundaries["joins"], 4)
        self.assertEqual(boundaries["timed_joins"], 2)
        self.assertEqual(boundaries["paired_tracks"], 2)
        self.assertEqual(boundaries["paired_timed_tracks"], 1)
        self.assertEqual(boundaries["near_join"], {"substitutions": 1, "deletions": 0,
                                                  "insertions": 1, "word_edits": 2})
        self.assertEqual(boundaries["paired"], {
            "unsegmented_word_edits": 1, "unsegmented_near_join_word_edits": 0,
            "segmentation_only_near_join_edits": 2, "unsegmented_only_near_join_edits": 0,
            "segmentation_only_word_edits": 2, "unsegmented_only_word_edits": 1,
            "identical_transcript_tracks": 1,
        })

    def test_compares_lexical_errors_without_timing_and_preserves_equal_count_differences(self):
        example = track("full-1", "one two")
        recognizer = FakeRecognizer({
            (example.audio, 2.0): transcript("red two", (0.1, 2.5), (2.0,)),
            (example.audio, None): transcript("one blue", (0.1, 2.5)),
        })

        report = evaluate_full_tracks(recognizer, [example], segment_seconds=2.0, compare_unsegmented=True)
        row = report["tracks"][0]
        self.assertEqual(row["word_edits"], 1)
        self.assertIsNone(row["segment_boundaries"]["near_join"])
        paired = row["segment_boundaries"]["paired"]
        self.assertFalse(paired["identical_transcript"])
        self.assertEqual(paired["unsegmented_word_edits"], 1)
        self.assertEqual(paired["segmentation_only_word_edits"], 1)
        self.assertEqual(paired["unsegmented_only_word_edits"], 1)
        self.assertIsNone(paired["segmentation_only_near_join_edits"])
        boundaries = report["aggregate"]["segment_boundaries"]
        self.assertEqual(boundaries["paired_tracks"], 1)
        self.assertEqual(boundaries["paired_timed_tracks"], 0)
        self.assertEqual(boundaries["paired"]["identical_transcript_tracks"], 0)
        self.assertIsNone(boundaries["paired"]["unsegmented_near_join_word_edits"])

    def test_shared_deletion_is_localized_from_reference_without_claiming_it_was_introduced(self):
        example = track("full-1", "one two", words=(
            TimedReferenceWord("one", 1.7, 1.9), TimedReferenceWord("two", 2.1, 2.2),
        ), instrumental_intervals=())
        recognizer = FakeRecognizer({
            (example.audio, 2.0): transcript("two", (2.1,), (2.0,)),
            (example.audio, None): transcript("two", (2.1,)),
        })

        report = evaluate_full_tracks(recognizer, [example], segment_seconds=2.0, compare_unsegmented=True)
        row = report["tracks"][0]
        self.assertEqual(row["segment_boundaries"]["near_join"]["deletions"], 1)
        self.assertEqual(row["segment_boundaries"]["paired"]["unsegmented_near_join_word_edits"], 1)
        self.assertEqual(row["segment_boundaries"]["paired"]["segmentation_only_near_join_edits"], 0)
        self.assertEqual(row["instrumental"]["words"], 0)
        self.assertIsNone(row["instrumental"]["words_per_minute"])

    def test_timing_breaks_alignment_ties_between_repeated_words(self):
        example = track("repeat", "oh oh", words=(
            TimedReferenceWord("oh", 1.9, 2.0), TimedReferenceWord("oh", 5.0, 5.1),
        ))
        recognizer = FakeRecognizer({
            (example.audio, 2.0): transcript("oh", (1.9,), (2.0, 4.0)),
            (example.audio, None): transcript("oh oh", (1.9, 5.0)),
        })

        report = evaluate_full_tracks(recognizer, [example], segment_seconds=2.0, compare_unsegmented=True)
        row = report["tracks"][0]
        self.assertEqual(row["wer"], 0.5)
        self.assertEqual(row["deletions"], 1)
        self.assertEqual(row["segment_boundaries"]["near_join"]["deletions"], 0)
        self.assertEqual(row["segment_boundaries"]["paired"]["segmentation_only_near_join_edits"], 0)

    def test_auxiliary_full_mixes_contribute_only_to_separate_instrumental_diagnostics(self):
        lexical = track("lyrics", "one two", instrumental_intervals=((4.0, 6.0),))
        diagnostic = FullTrack(
            track_id="diagnostic", song_id="diagnostic", artist_ids=("another",), lyric_id="another",
            recording_family_id="another", genre="rock", split="test", audio=Path("diagnostic.wav"),
            text="", duration_seconds=8.0, instrumental_intervals=((0.0, 2.0), (5.0, 6.0)),
            diagnostic_only=True,
        )
        results = {
            (lexical.audio, 20.0): transcript("one two", (0.1, 2.5), (2.0,)),
            (lexical.audio, None): transcript("one two", (0.1, 2.5)),
            (diagnostic.audio, 20.0): transcript("false regular false", (0.5, 3.0, 5.1), (2.0, 4.0)),
        }
        recognizer = FakeRecognizer(results)
        report = evaluate_full_tracks(recognizer, [diagnostic, lexical], compare_unsegmented=True)
        self.assertEqual(recognizer.calls, [(diagnostic.audio, 20.0), (lexical.audio, 20.0), (lexical.audio, None)])

        bank = report.pop("instrumental_diagnostics")
        expected_lexical = evaluate_full_tracks(FakeRecognizer(results), [lexical], compare_unsegmented=True)
        self.assertEqual(report, expected_lexical)
        self.assertEqual(bank["track_count"], 1)
        self.assertEqual(bank["duration_seconds"], 8.0)
        self.assertEqual(bank["instrumental"], {
            "intervals": 2, "intervals_with_words": 2, "seconds": 3.0, "words": 2,
            "interval_false_alarm_rate": 1.0, "words_per_minute": 40.0,
        })
        [score] = bank["tracks"]
        self.assertEqual(score["track_id"], diagnostic.track_id)
        self.assertEqual(score["instrumental"], bank["instrumental"])


if __name__ == "__main__":
    unittest.main()
