"""Export aligned, reviewed CSD excerpts for a local-only training diagnostic."""

import argparse
import csv
import math
import wave
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from chorustrace.data import normalize_text, write_jsonl

# Index ranges refer to source text/phonetic lines, including blank lines.
# Choose old verses with positive edition evidence rather than accepting every variant.
NURSERY = "https://www.gutenberg.org/cache/epub/32415/pg32415-images.html"
CAROLS = "https://hymncommons.org/hymn-commons-home/christmas-public-domain-carols/"
CSD = "https://zenodo.org/records/4785016"
TRACKS = {
    12: ("o-come-all-ye-faithful", "carol", CAROLS, [(0, 5), (6, 11)]),
    14: ("the-first-noel", "carol", CAROLS, None),
    16: (
        "grandfathers-clock",
        "folk",
        "https://www.loc.gov/resource/music.mussm-sm1876-7601869/",
        None,
    ),
    19: (
        "o-holy-night",
        "carol",
        "https://hymnary.org/text/o_holy_night_the_stars_are_brightly_shin",
        None,
    ),
    21: ("jingle-bells", "carol", "https://www.loc.gov/item/2023838255/", None),
    22: ("joy-to-the-world", "carol", CAROLS, None),
    23: (
        "auld-lang-syne",
        "folk",
        "https://www.scottishpoetrylibrary.org.uk/poem/auld-lang-syne/",
        None,
    ),
    24: ("little-bo-peep", "nursery-rhyme", NURSERY, None),
    28: ("london-bridge", "nursery-rhyme", NURSERY, [(0, 4)]),
    29: (
        "mary-had-a-little-lamb",
        "nursery-rhyme",
        "https://rpo.library.utoronto.ca/content/marys-lamb",
        None,
    ),
    30: (
        "we-wish-you-a-merry-christmas",
        "carol",
        "https://imslp.org/wiki/We_Wish_You_a_Merry_Christmas_(Warrell,_Arthur_Sydney)",
        [(0, 5)],
    ),
    31: (
        "muffin-man",
        "nursery-rhyme",
        "https://www.londonmuseum.org.uk/docklands/families/rhymes-in-time/the-muffin-man/",
        [(0, 4), (5, 9)],
    ),
    33: (
        "home-sweet-home",
        "folk",
        "https://citizen-dj.labs.loc.gov/loc-jukebox-folk-songs/remix/?itemId=jukebox-422372&itemStart=63000",
        None,
    ),
    36: (
        "pop-goes-the-weasel",
        "nursery-rhyme",
        "https://www.phrases.org.uk/meanings/pop-goes-the-weasel.html",
        [(0, 4)],
    ),
    37: (
        "rock-a-bye-baby",
        "nursery-rhyme",
        "https://findingaids.loc.gov/repositories/15/archival_objects/1009797",
        [(0, 4)],
    ),
    38: (
        "row-row-row-your-boat",
        "nursery-rhyme",
        "https://www.themorgan.org/music-manuscripts-and-printed-music/311191",
        [(0, 4)],
    ),
    42: (
        "twinkle-twinkle-little-star",
        "nursery-rhyme",
        "https://rpo.library.utoronto.ca/content/star",
        None,
    ),
    49: (
        "working-on-the-railroad",
        "folk",
        "https://en.wikisource.org/wiki/I%27ve_Been_Working_on_the_Railroad",
        [(0, 4)],
    ),
    50: (
        "when-you-and-i-were-young-maggie",
        "folk",
        "https://levysheetmusic.mse.jhu.edu/collection/130/128",
        None,
    ),
}


def clip_ranges(notes, grapheme_lines, phonetic_lines, selected):
    if len(grapheme_lines) != len(phonetic_lines):
        raise ValueError("grapheme and phonetic line counts differ")
    if [syllable for line in phonetic_lines for syllable in line] != [
        n["syllable"] for n in notes
    ]:
        raise ValueError("score syllables differ from phonetic annotation")
    note_at_line = [0]
    for line in phonetic_lines:
        note_at_line.append(note_at_line[-1] + len(line))
    if selected is None:
        selected = [(0, len(grapheme_lines))]
    for first, last in selected:
        stanza = []
        for index in range(first, last):
            if phonetic_lines[index]:
                if not normalize_text(grapheme_lines[index]):
                    raise ValueError("timed syllables have an empty grapheme label")
                start, end = note_at_line[index], note_at_line[index + 1]
                if (
                    stanza
                    and float(notes[end - 1]["end"])
                    - float(notes[stanza[0][0]]["start"])
                    > 25
                ):
                    yield stanza
                    stanza = []
                stanza.append((start, end, grapheme_lines[index]))
            elif stanza:
                yield stanza
                stanza = []
        if stanza:
            yield stanza


def read_annotations(z, recording):
    lyric = z.read(f"CSD/english/lyric/{recording}.txt").decode().splitlines()
    phonetic = [
        line.split()
        for line in z.read(f"CSD/english/txt/{recording}.txt").decode().splitlines()
    ]
    notes = list(
        csv.DictReader(z.read(f"CSD/english/csv/{recording}.csv").decode().splitlines())
    )
    return notes, lyric, phonetic


def prepare(archive: Path, root: Path) -> None:
    sources = []
    rights = []
    (root / "audio").mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        for number, (song, genre, lyric_source, selected) in TRACKS.items():
            rights.append(
                {
                    "rights_id": f"csd-{song}",
                    "source": CSD,
                    "recording_permission": "CC BY-NC-SA 4.0 research release by CSD creators.",
                    "lyrics_permission": f"Reviewed lines in traditional/public-domain lyrics ({lyric_source}).",
                    "training_allowed": True,
                    "checkpoint_release_allowed": False,
                }
            )
            for key in "ab":
                recording = f"en{number:03d}{key}"
                notes, lyric, phonetic = read_annotations(z, recording)
                # ZipFile independently checks every selected member's CRC from the official archive.
                raw = z.read(f"CSD/english/wav/{recording}.wav")
                audio = f"audio/csd-{recording}.wav"
                with wave.open(BytesIO(raw)) as source:
                    rate = source.getframerate()
                    previous_clip_end = 0
                    for index, stanza in enumerate(
                        clip_ranges(notes, lyric, phonetic, selected), 1
                    ):
                        start = stanza[0][0]
                        end = stanza[-1][1]
                        begin = float(notes[start]["start"])
                        finish = float(notes[end - 1]["end"])
                        previous = float(notes[start - 1]["end"]) if start else 0.0
                        following = (
                            float(notes[end]["start"])
                            if end < len(notes)
                            else source.getnframes() / rate
                        )
                        begin = max(
                            0.0, begin - min(0.12, max(0.0, begin - previous) / 2)
                        )
                        finish = min(
                            following,
                            finish + min(0.12, max(0.0, following - finish) / 2),
                        )
                        first_frame = max(previous_clip_end, math.floor(begin * rate))
                        last_frame = math.ceil(finish * rate)
                        if not 0 <= first_frame < last_frame <= source.getnframes():
                            raise ValueError(
                                f"invalid audio interval {recording}: {first_frame}:{last_frame}"
                            )
                        previous_clip_end = last_frame
                        clip = f"csd-{recording}-{index:02d}"
                        sources.append(
                            {
                                "song_id": song,
                                "recording_id": f"csd-{recording}",
                                "clip_id": clip,
                                "artist": "csd-anonymous-professional-singer",
                                "genre": genre,
                                "language": "en",
                                "annotation_verified": True,
                                "split": "train",
                                "alignment_verified": True,
                                "text": " ".join(line for _, _, line in stanza),
                                "start": first_frame / rate,
                                "end": last_frame / rate,
                                "vocal": audio,
                                "rights_id": f"csd-{song}",
                                "original_start_frame": first_frame,
                                "original_end_frame": last_frame,
                            }
                        )
                (root / audio).write_bytes(raw)
    write_jsonl(root / "sources.jsonl", sources)
    write_jsonl(root / "rights.jsonl", rights)
    seconds = sum(row["end"] - row["start"] for row in sources)
    print(
        f"{len(sources)} clips from {len(rights)} compositions, {seconds:.6f} seconds"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
