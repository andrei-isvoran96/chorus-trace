"""Build a local freeze manifest from the verified Jam-ALT and RWC inventories."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from chorustrace.data import normalize_text, read_jsonl, write_jsonl


JAM_REVIEW = "https://arxiv.org/html/2408.06370v1#S2"
JAM_RELEASE = "https://audioshake.github.io/jam-alt/"
RWC_RELEASE = "https://zenodo.org/records/18656623"


def _without_backing(text: str) -> str:
    lead, depth = [], 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced backing-vocal tags")
            if depth == 0:
                lead.append(" ")
        elif depth == 0:
            lead.append(char)
    if depth:
        raise ValueError("unbalanced backing-vocal tags")
    return "".join(lead)


def _lead_text(reviewed: str, original_words: list[dict]) -> str:
    lead = _without_backing(reviewed)
    digits = re.findall(r"\d+", lead)
    if not digits:
        return normalize_text(lead)
    if digits != ["29"]:
        raise ValueError("unverified spoken number in the reviewed lead text")
    marker = "numbertobecorroborated"
    tokens = normalize_text(re.sub(r"\b29\b", marker, lead)).split()
    if tokens.count(marker) != 1:
        raise ValueError("ambiguous spoken number in the reviewed lead text")
    position = tokens.index(marker)
    anchor = tokens[position - 3:position] + ["twenty", "nine"] + tokens[position + 1:position + 4]
    timed_tokens = normalize_text(" ".join(word["text"] for word in original_words)).split()
    if len(anchor) != 8 or sum(timed_tokens[i:i + 8] == anchor for i in range(len(timed_tokens))) != 1:
        raise ValueError("could not corroborate the spoken number against the original timed words")
    return " ".join(tokens[:position] + ["twenty", "nine"] + tokens[position + 1:])


def _identity(value: str) -> str:
    return "".join(character for character in normalize_text(value) if character.isalnum())


def _manifest_row(item: dict, audio: Path, destination: Path, reference: dict, **fields) -> dict:
    with audio.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"{audio} differs from the staged inventory")
    reference_path = destination / (audio.stem + ".reference.json")
    reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "track_id": item["track_id"], "artist_ids": [_identity(item["artist"])],
        "audio": os.path.relpath(audio, destination), "reference": reference_path.name,
        "split": "test", "full_track": True, "condition": "original_mix",
        "source_frames": item["frames"], "source_sample_rate": item["sample_rate"], **fields,
    }


def prepare(jam: Path, rwc: Path, destination: Path) -> Path:
    revised = {row["name"]: row for _, row in read_jsonl(jam / "metadata.jsonl") if row["language"] == "en"}
    earlier = {row["name"]: row for _, row in read_jsonl(jam / "original-metadata.jsonl") if row["language"] == "en"}
    jam_inventory = json.loads((jam / "inventory.json").read_text(encoding="utf-8"))
    jazz_inventory = json.loads((rwc / "inventory.json").read_text(encoding="utf-8"))
    pairs = {"jamalt:" + hashlib.sha256(name.encode()).hexdigest()[:16]: (row, earlier[name])
             for name, row in revised.items()}
    if len(pairs) != 20 or set(pairs) != {item["track_id"] for item in jam_inventory["tracks"]}:
        raise ValueError("the English reviewed release and staged inventory must match exactly")
    destination.mkdir(parents=True)
    rows = []
    for item in jam_inventory["tracks"]:
        review, original = pairs[item["track_id"]]
        if (review["artist"], review["title"], review["genre"]) != (item["artist"], item["title"], item["genre"]):
            raise ValueError("reviewed recording identity differs from the staged inventory")
        work = ":".join((_identity(item["artist"]), _identity(item["title"])))
        reference = {
            "text": _lead_text(review["text"], original["words"]), "complete_lead_vocal": True,
            "transcribed_by": "Jam-ALT published proficient-language reviser",
            "checked_by": "Jam-ALT published two separate lyric reviewers",
            "review_evidence": (f"{JAM_REVIEW}; pinned Jam-ALT v1.4.0 review, using explicit backing-vocal "
                                "marks to retain the lead and the original timed source for the one spoken number"),
        }
        rows.append(_manifest_row(
            item, jam / item["audio"], destination, reference,
            song_id=work, lyric_id=work, recording_family_id=work, genre=review["genre"], language="en",
            audio_source=item["source_url"],
            recording_permission=f"{item['recording_license']}, {jam_inventory['original_recording_license']}",
            lyrics_permission=f"Local benchmarking under track terms and the authors' published evaluation instructions, {JAM_RELEASE}",
        ))
    for item in jazz_inventory["tracks"]:
        work = _identity(item["title"].split("(", 1)[0])
        reference = {
            "checked_by": "RWC Jazz published original instrumentation catalog",
            "review_evidence": jazz_inventory["metadata_source"],
            "instrumental_intervals": item["instrumental_intervals"],
        }
        rows.append(_manifest_row(
            item, rwc / item["audio"], destination, reference, diagnostic_only=True,
            song_id=work, lyric_id=work, recording_family_id=work, genre="jazz",
            audio_source=jazz_inventory["source"], recording_permission=f"CC BY-NC 4.0, {RWC_RELEASE}",
        ))
    manifest = destination / "manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jam", type=Path)
    parser.add_argument("rwc", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(prepare(args.jam, args.rwc, args.destination))
