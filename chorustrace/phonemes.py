"""Training-only pronunciation supervision for licensed, aligned lyric clips."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .data import normalize_text
from .fulltrack_data import FullTrack


def _text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PhonemeSupervision:
    vocabulary: tuple[str, ...]
    targets: dict[str, tuple[int, ...]]
    sha256: str

    @property
    def classes(self) -> int:
        return len(self.vocabulary) + 1


def read_phoneme_supervision(path: Path, tracks: list[FullTrack]) -> PhonemeSupervision:
    source = path.read_bytes()
    payload = json.loads(source)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"{path}: expected version 1 phoneme supervision")
    vocabulary, examples = payload.get("vocabulary"), payload.get("examples")
    if (not isinstance(vocabulary, list) or not vocabulary
            or any(not isinstance(x, str) or not x.strip() for x in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError(f"{path}: expected unique nonempty phoneme tokens")
    if not isinstance(examples, dict) or not examples:
        raise ValueError(f"{path}: expected nonempty phoneme supervision examples")
    expected = {track.track_id: _text_hash(track.text) for track in tracks
                if track.split == "train" and not track.full_track and track.text}
    targets = {}
    for clip, annotation in examples.items():
        if clip not in expected:
            raise ValueError(f"{path}: unknown or ineligible training clip {clip!r}")
        if not isinstance(annotation, dict) or annotation.get("grapheme_sha256") != expected[clip]:
            raise ValueError(f"{path}: transcript checksum mismatch for {clip!r}")
        phones = annotation.get("phones")
        if (not isinstance(phones, list) or not phones
                or any(type(x) is not int or not 1 <= x <= len(vocabulary) for x in phones)):
            raise ValueError(f"{path}: invalid phoneme target for {clip!r}")
        targets[clip] = tuple(phones)
    return PhonemeSupervision(tuple(vocabulary), targets, hashlib.sha256(source).hexdigest())
