"""Verify candidate WAVs and prepare two blank local listening packets."""

import argparse
import json
import math
import shlex
from pathlib import Path

from chorustrace.data import read_jsonl, string_field
from chorustrace.fulltrack_data import _audio_signature


_COORDINATOR = """# Independent lead-vocal reference review

Give one pass directory to each of two people. Keep both passes independent and
private until submitted. Each `tracks.json` describes the same verified original
mixes. Paths point to the local WAVs. Keep the test assets, drafts and final
references accessible only to the evaluator.

Keep the original drafts. Compare every word, word boundary and instrumental
interval. Replay all disagreements against the verified WAV. Record the reviewed
WAV SHA-256 and the resolution for any disagreements. Leave tracks with unresolved
words or incomplete coverage out of the frozen set.

Write the reconciled reference separately in the template schema. In `text`, put
the words in the same order as `words`, with one space between tokens. Set
`complete_lead_vocal` to true after the full track is reconciled. Put the actual
transcriber and independent checker identities in `transcribed_by` and
`checked_by`. Record pointers to both original drafts and the reconciliation
notes in `review_evidence`. Start the test freeze only after completing this
process and checking the verified audio and rights evidence.
"""


_REVIEWER = """# Full-track listening pass

Before listening, match the file SHA-256 to `tracks.json` using the `checksum`
command. Use the `playback` command for the original WAV, or open the same
file in a local waveform editor.
Listen from the first frame through the end, including intros, breaks, fades and
outros. Replay ambiguous passages. Work from the audio without seeing the other
pass or any model output.

- Fill `words` with one object per audible lead word, with keys `word`, `start`
  and `end`. Record every repeated or changed word and audible lead-vocal ad-lib.
  Spell numbers as sung and keep contractions together. Exclude backing-only
  words. Join these tokens in order with one space in `text`.
- Use seconds relative to the first PCM frame, with millisecond precision. Mark
  where each word starts and ends in the original mix. Keep word spans in order
  with positive duration, inside the track duration, and without overlaps. Use
  a shared transition time if two words touch. Use the exact `duration_seconds`
  from the track index for a span that reaches the end of the WAV. Keep
  uncertain words or timing in separate timestamped notes for reconciliation.
- Fill `instrumental_intervals` with `[start, end]` pairs for passages with no
  singing or speech, including silent passages. Use the same full-track clock.
  Record each passage once, ordered with positive duration and no overlaps.
  Leave passages with backing vocals or other voices out of this interval list.
  Use an empty list after checking a track that has no instrumental passage.
- Enter your own identity in `transcribed_by`. Leave `checked_by` and
  `review_evidence` blank for reconciliation. Preserve your draft and notes.
"""


_BLANK_REFERENCE = {
    "text": None,
    "words": None,
    "instrumental_intervals": None,
    "complete_lead_vocal": False,
    "transcribed_by": None,
    "checked_by": None,
    "review_evidence": None,
}


def _inventory_rows(inventory: Path):
    if inventory.suffix.lower() == ".jsonl":
        yield from read_jsonl(inventory)
        return
    source = json.loads(inventory.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("tracks"), list):
        raise ValueError(f"{inventory}: expected an object with a tracks array")
    for index, row in enumerate(source["tracks"], start=1):
        location = f"{inventory}: track {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected an object")
        yield location, row


def _verified_tracks(inventory: Path) -> list[dict]:
    tracks, seen_ids, seen_pcm = [], set(), set()
    for location, row in _inventory_rows(inventory):
        audio = (inventory.parent / string_field(row, "audio", location)).resolve(strict=True)
        source = {
            "source_frames": row.get("source_frames", row.get("frames")),
            "source_sample_rate": row.get("source_sample_rate", row.get("sample_rate")),
        }
        duration, file_hash, pcm_hash = _audio_signature(audio, source, location)
        hash_key = "audio_sha256" if "audio_sha256" in row else "sha256"
        if file_hash != string_field(row, hash_key, location):
            raise ValueError(f"{location}: WAV SHA-256 does not match inventory")
        expected = row.get("duration_seconds", duration)
        if type(expected) not in (int, float) or not math.isclose(
            expected, duration, rel_tol=0, abs_tol=0.5 / source["source_sample_rate"],
        ):
            raise ValueError(f"{location}: WAV duration does not match inventory")
        key = f"track-{pcm_hash[:16]}"
        track_id = string_field(row, "track_id", location) if "track_id" in row else key
        if track_id.casefold() in seen_ids or pcm_hash in seen_pcm:
            raise ValueError(f"{location}: duplicate track ID or PCM audio")
        seen_ids.add(track_id.casefold())
        seen_pcm.add(pcm_hash)
        quoted_audio = shlex.quote(str(audio))
        tracks.append(source | {
            "track_id": track_id,
            "audio": str(audio),
            "audio_sha256": file_hash,
            "pcm_sha256": pcm_hash,
            "duration_seconds": duration,
            "reference": f"references/{key}.json",
            "checksum": f"shasum -a 256 {quoted_audio}",
            "playback": f"ffplay -hide_banner -autoexit -showmode 1 -i {quoted_audio}",
        })
    if not tracks:
        raise ValueError(f"{inventory}: no WAVs to review")
    return tracks


def prepare(inventory: Path, output: Path) -> Path:
    """Create separate, blank packets after all WAVs pass the inventory checks."""
    tracks = _verified_tracks(inventory)
    output.mkdir(mode=0o700, parents=True)
    (output / "README.md").write_text(_COORDINATOR, encoding="utf-8")
    track_index = json.dumps(tracks, indent=2) + "\n"
    blank = json.dumps(_BLANK_REFERENCE, indent=2) + "\n"
    for label in ("pass-a", "pass-b"):
        directory = output / label
        (directory / "references").mkdir(parents=True)
        (directory / "README.md").write_text(_REVIEWER, encoding="utf-8")
        (directory / "tracks.json").write_text(track_index, encoding="utf-8")
        for track in tracks:
            (directory / track["reference"]).write_text(blank, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        print(prepare(args.inventory, args.output))
    except (OSError, ValueError) as error:
        parser.error(str(error))
