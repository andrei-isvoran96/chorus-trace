# Twenty-song MUSDB18 experiment

ChorusTrace trained from random initialization on **162 excerpts from 20 released recordings**, totaling 19 minutes 17 seconds of distinct lead-vocal audio. The test set has **39 excerpts from six different artists**, totaling 4 minutes 9 seconds. Test songs come from the official test partition and are also artist-disjoint from the training recordings. The hip-hop selections include rapped vocals.

## Recordings

| Genre | Training songs | Excerpts |
| --- | --- | ---: |
| Hip-hop | ANiMAL — Clinic A, Grants — PunchDrunk, Lushlife — Toynbee Suite | 13 |
| Rock | AvaLuna — Waterduct, BigTroubles — Phantom, The Districts — Vermont, The So So Glos — Emergency | 38 |
| Pop | A Classic Education — NightOwl, Dreamers Of The Ghetto — Heavy Love, Secret Mountains — High Horse, Snowmine — Curfews, Sweet Lights — You Let Me Down | 39 |
| Electronic | Skelpolu — Human Mistakes, Triviul — Angelsaint | 18 |
| Folk / blues | Clara Berry And Wooldog — Waltz For My Victims, Hezekiah Jones — Borrowed Heart | 16 |
| Jazz | Matthew Entwistle — Dont You Ever, Patrick Talbot — Set Me Free | 18 |
| Country | Angela Thomas Wade — Milk Cow Blues | 10 |
| Metal | Hollow Ground — Left Blind | 10 |

| Genre | Held-out songs | Excerpts |
| --- | --- | ---: |
| Hip-hop | M.E.R.C. Music — Knockout, Side Effects Project — Sing With Me | 9 |
| Rock | Nerve 9 — Pray For The Rain | 9 |
| Pop | Lyndsey Ollard — Catching Up | 6 |
| Electronic | Girls Under Glass — We Feel Alright | 5 |
| Folk / blues | Cristina Vane — So Easy | 10 |

The genre labels are broad groupings for this experiment. The original releases and recording library identify the hip-hop tracks. [Cambridge Music Technology](https://www.cambridge-mt.com/ms3/mtk/) · [Grant$](https://weathervanemusic.org/shakingthrough/grants) · [Lushlife](https://weathervanemusic.org/shakingthrough/lushlife) · [Side Effects Project](https://preslindavis.bandcamp.com/track/sing-with-me)

The release notes describe `NightOwl` and `Curfews` as indie pop, `You Let Me Down` as power pop, and `Emergency` as punk. [A Classic Education](https://weathervanemusic.org/shakingthrough/aclassiceducation) · [Snowmine](https://weathervanemusic.org/shakingthrough/snowmine) · [Sweet Lights](https://weathervanemusic.org/shakingthrough/sweetlights) · [The So So Glos](https://weathervanemusic.org/shakingthrough/thesosoglos)

The [selection](musdb18_genre20.jsonl) records exact IDs, genres, splits, and timestamps. It uses unstarred single-lead class `a` annotations with at least two normalized words. Excerpts are 3–12 seconds and do not overlap within a song. Selections use the first 160 seconds, with longer prefixes for the rapped sections of `Toynbee Suite` and for `Catching Up`. Where more than ten eligible intervals remained, the marks sample ten evenly by ordinal index. The selections were fixed before fitting or scoring.

Each excerpt has seven views with the same lead waveform, including instrumental-only data for measuring false alarms. Checks found zero sample lag between the original mix and the sum of stems in 77 sampled windows across the 26 tracks. After PCM quantization, the largest deviation from a target accompaniment level was 0.00011 dB across all 201 excerpts.

The five-song and 20-song checkpoints share the architecture, seed, optimizer settings, and 30-epoch schedule. The runs differ in repertoire, excerpt counts, training encodings (HQ WAV versus decoded AAC), and optimizer updates (3,150 versus 34,020). Common-holdout scores compare the resulting checkpoints. The genre breakdown describes this small test set, where four genres have one song each.

## Reproduce

This run used the original MUSDB18 AAC stems decoded to PCM16 via FFmpeg 8.1.1. The mixture came from the original mix stream. The PCM prefixes, selected official annotations, WAV hashes, and alignment checks are under `data/musdb18_genre20` in the measured workspace. Training and inference used Python 3.12.13 and NumPy 2.5.3. The upstream multitrack source calls for contributor permission for AI training. [MUSDB18 audio and educational terms](https://zenodo.org/records/1117372) · [MUSDB18 partial lyrics](https://zenodo.org/records/3989267) · [Cambridge Music Technology terms](https://www.cambridge-mt.com/ms3/mtk-faq/)

To rebuild the local PCM inputs from the archives, save `musdb18.zip`, `train_lyrics.zip`, and `test_lyrics.zip` from those records in `data/musdb18_genre20`. FFmpeg is needed for this extraction step. The commands decode only the prefixes specified by the marks. The measured run acquired those prefixes via ZIP byte ranges.

```sh
python - <<'PY'
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

root = Path("data/musdb18_genre20")
marks = Path("benchmarks/musdb18_genre20.jsonl")
ends = {}
for line in marks.read_text().splitlines():
    row = json.loads(line)
    key = row["split"], row["song_id"]
    ends[key] = max(ends.get(key, 0), row["end"])
with ZipFile(root / "musdb18.zip") as audio:
    for (split, song), end in sorted(ends.items()):
        output = root / "audio" / split / song
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as temporary:
            member = audio.extract(f"{split}/{song}.stem.mp4", temporary)
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", member]
            for index, stem in enumerate(("mixture", "drums", "bass", "other", "vocals")):
                command += ["-map", f"0:a:{index}", "-t", str(end), "-c:a", "pcm_s16le", str(output / f"{stem}.wav")]
            subprocess.run(command, check=True)
for split in ("train", "test"):
    with ZipFile(root / f"{split}_lyrics.zip") as lyrics:
        for kind, song in ends:
            if kind == split:
                lyrics.extract(f"{song}.txt", root / "lyrics" / f"{split}_lyrics")
PY
```

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
chorustrace prepare-musdb18 data/musdb18_genre20/audio data/musdb18_genre20/lyrics \
  benchmarks/musdb18_genre20.jsonl runs/musdb18-genre20
chorustrace train runs/musdb18-genre20/manifest.jsonl runs/musdb18-genre20/baseline.npz \
  --epochs 30 --hop-size 320 --seed 0
chorustrace evaluate runs/musdb18-genre20/baseline.npz runs/musdb18-genre20/manifest.jsonl \
  --held-out --samples 39 > runs/musdb18-genre20/evaluate.json
python benchmarks/score_genres.py runs/musdb18-genre20/baseline.npz \
  runs/musdb18-genre20/manifest.jsonl benchmarks/musdb18_genre20.jsonl \
  > runs/musdb18-genre20/genres.json
```

Compare with the five-song checkpoint from the [initial experiment](../README.md#real-audio-benchmark). One command scores it on the new holdout. The other scores the new checkpoint on the original holdout.

```sh
python benchmarks/score_genres.py runs/musdb18/baseline.npz \
  runs/musdb18-genre20/manifest.jsonl benchmarks/musdb18_genre20.jsonl \
  > runs/musdb18-genre20/five_song_genres.json
chorustrace evaluate runs/musdb18-genre20/baseline.npz runs/musdb18/manifest.jsonl \
  --held-out --samples 9 > runs/musdb18-genre20/original_holdout.json
```

## Results

The new holdout has 432 reference words and 2,018 characters per vocal condition. WER and CER are summed edit counts divided by the corresponding reference total. Both models scored the same WAVs in this comparison.

The checkout includes [aggregate scores](results/musdb18_genre20.json), [scores by genre](results/musdb18_genre20_by_genre.json), and [scores from the original holdout](results/musdb18_genre20_original_holdout.json).

| New holdout condition | Five-song WER / CER | Twenty-song WER / CER |
| --- | ---: | ---: |
| Isolated vocal | 102.3% / 85.1% | 99.1% / 77.2% |
| Original mix | 103.5% / 87.4% | 98.8% / 82.0% |
| Accompaniment −12 dB | 103.9% / 86.2% | 98.8% / 78.7% |
| Accompaniment −6 dB | 104.6% / 86.8% | 99.3% / 80.4% |
| Accompaniment 0 dB | 104.6% / 87.1% | 99.5% / 82.2% |
| Accompaniment +6 dB | 104.6% / 87.8% | 98.8% / 82.1% |
| Instrumental clips with word output | 28/39 | 23/39 |

| Held-out genre | Excerpts | Twenty-song isolated WER / CER | Twenty-song original mix WER / CER | Instrumental clips with word output |
| --- | ---: | ---: | ---: | ---: |
| Hip-hop | 9 | 98.0% / 75.3% | 98.0% / 79.5% | 7/9 |
| Rock | 9 | 100.0% / 78.6% | 100.0% / 80.8% | 4/9 |
| Pop | 6 | 100.0% / 80.8% | 100.0% / 88.5% | 1/6 |
| Electronic | 5 | 100.0% / 78.3% | 97.1% / 84.9% | 5/5 |
| Folk / blues | 10 | 100.0% / 78.3% | 100.0% / 84.1% | 6/10 |

The original-mix model output was nonempty on all 39 excerpts, with frequent incorrect words. These are decoded examples from the saved 20-song checkpoint.

| Recording and interval | Isolated prediction | Original mix prediction |
| --- | --- | --- |
| M.E.R.C. Music — Knockout, 57–63 s | `minsahi nwy ih t tbn s tihh` | `iecs i nw t t as hti` |
| Lyndsey Ollard — Catching Up, 57–64 s | `mn oce sr` | `olone` |
| Nerve 9 — Pray For The Rain, 17–25 s | `s as yra` | `so ss r` |

On the original nine-excerpt HQ holdout, isolated CER fell from 81.2% to 78.2% and original-mix CER from 87.8% to 85.1%. Original-mix WER rose from 98.7% to 100.0%. Instrumental false alarms fell from 5/9 to 3/9. The larger run improved character accuracy on both holdouts. Word error remains near 100%, and the model still produces many false alarms.
