# ChorusTrace

ChorusTrace trains a local recognizer for short English singing clips with one lead vocal. It reads integer PCM WAV, extracts NumPy log-mel features, and fits a small dilated temporal network with handwritten backpropagation, character CTC, and Adam. Greedy decoding uses the learned frame scores. Training starts from random weights. Transcription needs a WAV and a saved checkpoint and runs offline.

## Install and use

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Python 3.10+ and NumPy are the runtime requirements. `python -m chorustrace` also works from the checkout once NumPy is installed. To install offline, put NumPy and Setuptools wheels in a local wheelhouse and use `python -m pip install --no-index --find-links wheelhouse -e .`.

A training or evaluation manifest is UTF-8 JSON Lines. WAV paths are relative to the manifest. This example uses a separate manifest for each split. For one combined file, put a `split` of `train`, `validation`, or `test` on every row. All rows of a song must share its split. Use empty `text` for instrumental clips.

```json
{"audio":"wav/clip.wav","text":"Hello moon","song_id":"song-01"}
```

```sh
chorustrace train train.jsonl runs/model.npz --epochs 30 --seed 0
chorustrace transcribe runs/model.npz wav/clip.wav
chorustrace evaluate runs/model.npz test.jsonl --held-out
```

Save an editable draft with `chorustrace transcribe runs/model.npz wav/clip.wav > draft.txt`.
Train the longer-context model with `--architecture long-context`. If you have a
calibration JSON for that exact checkpoint, pass `--abstention calibration.json`
to return an empty draft when lyric confidence is at or below its threshold.

See the [benchmark report](benchmarks/english_lead_benchmark.md) for rights-reviewed
data preparation, artist-disjoint splits, and reproduction commands.

```python
from chorustrace import Recognizer, TrainingConfig, read_manifest, train

model = train(read_manifest("train.jsonl"), TrainingConfig(epochs=30, seed=0))
model.save("runs/model.npz")
print(Recognizer.load("runs/model.npz").transcribe("wav/clip.wav"))
```

WAV input supports 8-, 16-, 24-, and 32-bit integer PCM and downmixes channels to mono. Labels are lowercased and reduced to English letters, spaces, and internal apostrophes. Feature normalization uses training frames and is stored with the model. The CTC tests include a hand-checkable case where the three-frame `a, blank, a` path has probability `1/8` and is the only way to produce `aa`.

## Offline smoke test and synthetic benchmark

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
chorustrace prepare-synthetic runs/synthetic
chorustrace train runs/synthetic/smoke.jsonl runs/synthetic/smoke.npz \
  --epochs 80 --learning-rate 0.008 --hop-size 320 --seed 0
chorustrace transcribe runs/synthetic/smoke.npz \
  "$(python -c 'import json; print("runs/synthetic/" + json.load(open("runs/synthetic/smoke.jsonl"))["audio"])')"
chorustrace train runs/synthetic/manifest.jsonl runs/synthetic/baseline.npz \
  --epochs 30 --hop-size 320 --seed 0
chorustrace evaluate runs/synthetic/baseline.npz runs/synthetic/manifest.jsonl \
  --held-out --samples 9 > runs/synthetic/evaluate.json
chorustrace chorus runs/synthetic/baseline.npz runs/synthetic/manifest.jsonl \
  runs/synthetic/repeats.jsonl --held-out > runs/synthetic/chorus_mix.json
chorustrace chorus runs/synthetic/baseline.npz runs/synthetic/manifest.jsonl \
  runs/synthetic/repeats.jsonl --held-out --condition vocal > runs/synthetic/chorus_vocal.json
```

The smoke transcription is `hello moon`, including both doubled letters. It measures memorization and checkpoint reload. The synthetic benchmark uses 24 training excerpts from six songs and nine held-out excerpts from three songs. The measured vocal WER/CER was 113.8%/52.4% and original-mix WER/CER was 120.7%/60.3%. Instrumental false alarms were 0/9. The fixtures have procedural voices and accompaniment.

The measured [synthetic smoke checkpoint](benchmarks/checkpoints/synthetic_smoke.npz), [synthetic baseline checkpoint](benchmarks/checkpoints/synthetic_baseline.npz), and [benchmark results](benchmarks/results/) are included in the checkout.

## Real-audio benchmark

The follow-up experiment trained on 20 released songs across eight genres and tested six held-out artists. [Twenty-song experiment](benchmarks/musdb18_genre20.md)

The measured experiment used the MUSDB18-HQ WAV release and MUSDB18 partial lyrics annotations. The marks in `benchmarks/` select exact intervals from their official train and test sets. The main benchmark uses unstarred class `a` intervals for a single singer. The separate chorus check uses three manually selected class `b` intervals with multiple voices singing the same phonemes. [MUSDB18 terms](https://zenodo.org/records/1117372) · [MUSDB18-HQ WAVs](https://zenodo.org/records/3338373) · [Partial lyrics](https://zenodo.org/records/3989267)

The dataset record describes educational use and refers to track-specific rights. Cambridge Music Technology's current upstream FAQ calls for contributor permission for AI training on its multitracks. The lyrics extension also identifies the composers as the rights holders for the underlying lyrics. Use audio and annotations you have permission to train on. The recordings, reference lyrics, and real-audio checkpoints are local experiment files. [Cambridge Music Technology terms](https://www.cambridge-mt.com/ms3/mtk-faq/)

If `data/musdb18/audio` and `data/musdb18/lyrics` are present, use the prepared subset and skip archive extraction. The measured run used the original PCM through each selected interval, obtained via byte ranges from the HQ ZIP. To build the same inputs from the official archives, save `musdb18hq.zip`, `train_lyrics.zip`, and `test_lyrics.zip` from the records above in `data/musdb18/`. The full HQ ZIP is 22.7 GB. Extract just the required songs and stems as follows.

```sh
python - <<'PY'
import json
from pathlib import Path
from zipfile import ZipFile

root = Path("data/musdb18")
with ZipFile(root / "musdb18hq.zip") as archive:
    for selection, stems in [
        ("musdb18_subset", ("vocals", "mixture", "drums", "bass", "other")),
        ("musdb18_chorus", ("mixture",)),
    ]:
        marks = map(json.loads, Path(f"benchmarks/{selection}.jsonl").read_text().splitlines())
        tracks = sorted({(row["split"], row["song_id"]) for row in marks})
        for split, song in tracks:
            for stem in stems:
                archive.extract(f"{split}/{song}/{stem}.wav", root / "audio")
for split in ("train", "test"):
    with ZipFile(root / f"{split}_lyrics.zip") as archive:
        archive.extractall(root / "lyrics" / f"{split}_lyrics")
PY
```

These commands render the WAV conditions, train a single model on all training conditions, and score held-out songs. The smoke manifest selects the four-second isolated excerpt at 8–12 s in `Music Delta - Hendrix`.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
chorustrace prepare-musdb18 data/musdb18/audio data/musdb18/lyrics \
  benchmarks/musdb18_subset.jsonl runs/musdb18
python - <<'PY'
import json
from pathlib import Path

root = Path("runs/musdb18")
rows = map(json.loads, (root / "manifest.jsonl").read_text().splitlines())
row = next(r for r in rows if (r["song_id"], r["clip_id"], r["condition"]) ==
           ("Music Delta - Hendrix", "8-12", "vocal"))
(root / "smoke.jsonl").write_text(json.dumps(row) + "\n")
PY
chorustrace train runs/musdb18/smoke.jsonl runs/musdb18/smoke.npz \
  --epochs 120 --learning-rate 0.008 --hop-size 320 --seed 0
chorustrace transcribe runs/musdb18/smoke.npz \
  "$(python -c 'import json; print("runs/musdb18/" + json.load(open("runs/musdb18/smoke.jsonl"))["audio"])')"
chorustrace train runs/musdb18/manifest.jsonl runs/musdb18/baseline.npz \
  --epochs 30 --hop-size 320 --seed 0
chorustrace evaluate runs/musdb18/baseline.npz runs/musdb18/manifest.jsonl \
  --held-out --samples 9 > runs/musdb18/evaluate.json
```

### Held-out results

Measured with Python 3.12.13 and NumPy 2.5.3. The real smoke clip matched its reference exactly after reloading. The baseline trained on five songs with 15 unique excerpts totaling 80 seconds. Each excerpt supplied seven WAV views, including its instrumental-only view. The held-out set has three songs, nine distinct excerpts, and 58 seconds of singing per condition. `--held-out` checks song IDs against the checkpoint.

For each excerpt the isolated and mixed views use the same vocal waveform. Each controlled mix scales the sum of drums, bass, and other stems to the indicated RMS dB relative to the vocal. The original mixture uses the supplied mix stem. One shared headroom gain per excerpt prevents clipping. Instrumental clips use the matched accompaniment stem sum at its original level before that gain.

| Held-out MUSDB18-HQ condition | WER | CER |
| --- | ---: | ---: |
| Isolated vocal | 105.1% | 81.2% |
| Original mix | 98.7% | 87.8% |
| Accompaniment −12 dB | 107.7% | 85.1% |
| Accompaniment −6 dB | 105.1% | 86.2% |
| Accompaniment 0 dB | 98.7% | 87.3% |
| Accompaniment +6 dB | 100.0% | 89.8% |

Rates divide the total edits by 78 reference words or 362 reference characters, including spaces. Insertions can push WER over 100%. Five of nine instrumental clips produced nonempty word output. The 0–5 s excerpt from `Carlos Gonzalez - A Place For Us` produced `oas'sms c` in isolation and `oasms s lle` in the original mix. The baseline has high error across all conditions. On this small set, character error rose with accompaniment level in the controlled mixes.

### Marked chorus repeats

```sh
chorustrace prepare-musdb18 data/musdb18/audio data/musdb18/lyrics \
  benchmarks/musdb18_chorus.jsonl runs/musdb18-chorus --mix-only
chorustrace chorus runs/musdb18/baseline.npz runs/musdb18-chorus/manifest.jsonl \
  benchmarks/musdb18_repeats.jsonl --held-out > runs/musdb18-chorus/report.json
```

Each JSONL mark has `song_id`, `clip_a`, and `clip_b`. Clips refer to `clip_id` in the requested condition, which defaults to `original_mix`. The experiment finds a monotone dynamic time warping path from frame posterior distances, including blank. With each take as the anchor in turn, it averages aligned partner frames and mixes those probabilities 50/50 with the anchor before greedy decoding. Each output is scored against that take's own reference. An optional `changed_lyric` phrase unique to one reference reports whether pooling erases it or introduces it in the other take.

| Chorus pair | Independent WER / CER | Pooled WER / CER |
| --- | ---: | ---: |
| Real original mix, same lyric | 100.0% / 82.8% | 100.0% / 82.8% |
| Real original mix, changed lyric | 136.4% / 81.5% | 100.0% / 72.2% |
| Synthetic original mix, same lyric | 116.7% / 43.3% | 116.7% / 40.0% |
| Synthetic original mix, changed lyric | 100.0% / 46.4% | 83.3% / 50.0% |

The annotation for the real changed take omits an opening `and`. Both of its predictions omitted `and`, as did the independent prediction on the canonical take. The synthetic changed take substitutes `today` for `tonight`. Its original-mix prediction shifted from `e sogy` to `e e tonght`. The recognizer missed `today` independently, so preservation of a correctly recognized change remains unverified on audio. A controlled posterior test demonstrates that pooling can turn an independently correct `oh` into the other take's `ah`.

## Prepare your own paired benchmark

Provide a JSONL source manifest like the line below. All paths in one row must share a sample rate, duration, and time origin. `accompaniment` can be one WAV or a list of stems to sum. `start` and `end` in seconds are optional for pre-cut clips. Use a distinct `clip_id` per excerpt within each song.

```json
{"song_id":"song-01","clip_id":"verse-01","split":"train","text":"Hello moon","vocal":"vocals.wav","mix":"mixture.wav","accompaniment":["bass.wav","drums.wav","other.wav"],"start":8,"end":12}
```

```sh
chorustrace prepare sources.jsonl runs/custom --levels -12 -6 0 6
```

The feature pipeline, network, CTC implementation, and decoder live in `chorustrace/`. The CTC recurrence follows Graves et al., *Connectionist Temporal Classification*, 2006. [Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
