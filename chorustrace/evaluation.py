"""Character and word error rates for short-clip transcription."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .data import Example, normalize_text
from .recognizer import Recognizer


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    previous = list(range(len(prediction) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, predicted_item in enumerate(prediction, start=1):
            current.append(min(
                previous[column] + 1,
                current[-1] + 1,
                previous[column - 1] + (reference_item != predicted_item),
            ))
        previous = current
    return previous[-1]


@dataclass
class ErrorCounts:
    reference_words: int = 0
    word_edits: int = 0
    reference_characters: int = 0
    character_edits: int = 0

    def add(self, reference: str, prediction: str) -> None:
        words = reference.split()
        self.reference_words += len(words)
        self.word_edits += edit_distance(words, prediction.split())
        self.reference_characters += len(reference)
        self.character_edits += edit_distance(reference, prediction)

    def report(self) -> dict:
        return {
            "wer": self.word_edits / self.reference_words if self.reference_words else None,
            "cer": self.character_edits / self.reference_characters if self.reference_characters else None,
            "word_edits": self.word_edits, "reference_words": self.reference_words,
            "character_edits": self.character_edits, "reference_characters": self.reference_characters,
        }


def evaluate(recognizer: Recognizer, examples: Sequence[Example], samples: int = 3) -> dict:
    """Return micro-averaged error rates per condition and counts for empty references."""
    if samples < 0:
        raise ValueError("sample count must be nonnegative")
    sections = {}
    errors = defaultdict(ErrorCounts)
    for example in examples:
        prediction = recognizer.transcribe(example.audio)
        reference = normalize_text(example.text)
        section = sections.setdefault(example.condition, {
            "clips": 0, "voiced_clips": 0, "instrumental_clips": 0,
            "instrumental_with_words": 0, "samples": [],
        })
        section["clips"] += 1
        if reference:
            section["voiced_clips"] += 1
            errors[example.condition].add(reference, prediction)
        else:
            section["instrumental_clips"] += 1
            section["instrumental_with_words"] += bool(prediction.split())
        if len(section["samples"]) < samples:
            section["samples"].append({
                "song_id": example.song_id, "clip_id": example.clip_id,
                "reference": reference, "prediction": prediction,
            })

    for condition, section in sections.items():
        section.update(errors[condition].report())
        instrumental = section["instrumental_clips"]
        section["instrumental_word_rate"] = section["instrumental_with_words"] / instrumental if instrumental else None
    return {"songs": len({example.song_id for example in examples}), "conditions": sections}
