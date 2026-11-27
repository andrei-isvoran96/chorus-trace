# English lead-vocal benchmark

ChorusTrace fits an offline NumPy CTC recognizer from random initialization. The experiments use only the audio and its time-aligned training labels. The decoder emits plain text via greedy character CTC. The optional confidence gate can return an empty draft.

## Held-out real vocals

The reviewed Vocadito subset contains **225.173605 seconds (0.062548224 h)** of unaccompanied English singing from 10 singer IDs and 10 compositions. It covers folk, carols, nursery rhymes, and Tin Pan Alley songs. The excerpt mean pitches span MIDI 49–61. Its 11 recordings provide the following artist-disjoint splits.

| Split | Vocal clips | Singer IDs | Compositions | Seconds | Hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 8 | 7 | 7 | 179.711791 | 0.049919942 |
| Validation | 1 | 1 | 1 | 20.280000 | 0.005633333 |
| Test | 2 | 2 | 2 | 25.181814 | 0.006994948 |

The train split fell short of the 1-, 5-, and 10-hour rungs. Both architectures used the entire cleared training split and the same validation clip. These results give one data point per architecture, so the benefit of adding data remains unmeasured.

| Model | Steps | Fit time | Training WER / CER | Validation WER / CER |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 240 | 7.52 s | 94.55% / 66.18% | 100.00% / 81.82% |
| Long context | 240 | 7.99 s | 96.62% / 74.04% | 96.43% / 77.69% |

Long context won validation by one word edit (27/28 versus 28/28) and five character edits (94/121 versus 99/121). This comparison uses one validation singer. Its saved checkpoint was selected before the single test run and reloaded in a new process. The real corpus has no original mixes or accompaniment stems, so this checkpoint has no calibrated threshold.

| Usability target | Status | Held-out result |
| --- | --- | --- |
| Isolated vocal WER ≤25% | **Missed** | 100.00% WER (35/35), 80.40% CER (160/199), 2 clips |
| Original mix WER ≤35% | **Unmeasured** | 0 matched mixes |
| Instrumental false alarms <5% | **Unmeasured** | 0 instrumentals |

| Isolated test genre | Clips | Singers | WER | CER | Drafts returned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Folk | 1 | 1 | 100.00% (19/19) | 81.90% (86/105) | 1 |
| Nursery rhyme | 1 | 1 | 100.00% (16/16) | 78.72% (74/94) | 1 |

The folk reference begins `oh my darlin oh my darlin`. The full draft was `t sh h h aes a e s se a`. The nursery-rhyme reference begins `happy birthday to you`. Its draft was `i e aee e a e ae e e n asn n whane e`. Both require rewriting.

The selected checkpoint has SHA-256 `598dfa59b66e3d954dc0e888d2110d56d442bd619d5beb1d0927736757e22fd2`. The frozen manifest hash is `3af7310ee3dea19cc8a8b0517ea0cafac78ae0db74c78735f11d2a5c359c7b54`. The curve JSON files record optimizer losses and reference counts. The test JSON contains both full sample predictions. A separate transcription process with networking denied reproduced the validation draft byte for byte. [Checkpoint](checkpoints/vocadito_long_context.npz) · [Vocadito attribution](../docs/data-rights.md#vocadito-english-subset)

### Reproduce the real benchmark

Run from the checkout with the virtual environment activated. The acquisition step uses Zenodo. Preparation, fitting, transcription, and scoring then run offline. Preparation and fitting expect new output directories.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
DATA=data/cleared/verification-vocadito
REAL=runs/verification-vocadito

python -m scripts.acquire_vocadito "$DATA"
python -m scripts.prepare_lyrics freeze "$DATA/sources.jsonl" "$DATA/rights.jsonl" \
  "$REAL/prepared" --seed 0 --max-seconds 32
python -m scripts.prepare_lyrics verify "$REAL/prepared"
python -m scripts.learning_curve "$REAL/prepared" "$REAL/curve-baseline" --architecture baseline
python -m scripts.learning_curve "$REAL/prepared" "$REAL/curve-long-context" --architecture long-context

CHECKPOINT="$REAL/curve-long-context/available.npz"
VALIDATION_WAV=$(python -c 'import json, sys; from pathlib import Path; p=Path(sys.argv[1]); rows=map(json.loads,p.read_text().splitlines()); print(p.parent / next(r["audio"] for r in rows if r["split"] == "validation" and r["condition"] == "vocal"))' "$REAL/prepared/manifest.jsonl")
python -m chorustrace transcribe "$CHECKPOINT" "$VALIDATION_WAV" > "$REAL/validation-draft.txt"
python -m scripts.benchmark_lyrics "$REAL/prepared" "$CHECKPOINT" --split test > "$REAL/test.json"
```

### Cleared real-audio smoke test

The baseline memorized all 23 words and 110 characters of the shortest training clip (8.71 s) and recovered them after checkpoint reload. This one-clip smoke test uses a separate checkpoint.

```sh
SMOKE_WAV=$(python - <<'PY'
import os
from pathlib import Path
from chorustrace.data import read_jsonl, write_jsonl

root = Path('runs/verification-vocadito')
prepared = root / 'prepared'
row = min((r for _, r in read_jsonl(prepared / 'manifest.jsonl')
           if r['split'] == 'train' and r['condition'] == 'vocal'), key=lambda r: r['duration_seconds'])
audio = prepared / row['audio']
write_jsonl(root / 'real-smoke.jsonl', [dict(row, audio=os.path.relpath(audio, root))])
print(audio)
PY
)
python -m chorustrace train "$REAL/real-smoke.jsonl" "$REAL/real-smoke.npz" \
  --epochs 220 --learning-rate 0.008 --hop-size 320 --seed 0
python -m chorustrace transcribe "$REAL/real-smoke.npz" "$SMOKE_WAV" > "$REAL/real-smoke-draft.txt"
python -m chorustrace evaluate "$REAL/real-smoke.npz" "$REAL/real-smoke.jsonl" \
  --split train > "$REAL/real-smoke-report.json"
```

## Additional training-only corpus

The reviewed English subset of the Children's Song Dataset adds 140 isolated clips from 38 renditions of 19 compositions. Its single professional singer accounts for **2,158.807415 seconds (0.599668726 h)**. Vocadito uses pseudonymous singer IDs, so cross-corpus identity exclusion could not be verified. CSD was fitted separately under its noncommercial grant. The CSD checkpoint stays local and its scores cover the training clips from that one singer. [Source and rights review](../docs/data-rights.md)

The baseline ran 4,200 optimizer steps in 50.93 seconds at 30 epochs. A fresh process reloaded the checkpoint and measured 94.67% training WER (2,560/2,704) and 54.99% training CER (7,390/13,439). Together the two reviewed datasets provided 0.649588668 hours for fitting across the separate runs.

Place the official archive at `data/cleared/csd/CSD.zip`. The release MD5 for the full archive is `4c586caff29ca05264cf7223cad1b8a2`. The importer checks each selected ZIP member while reading it. The experiment archive was assembled by byte range and its selected members passed their CRC checks. [Official release](https://zenodo.org/records/4785016)

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
DATA=data/cleared/verification-csd
OUT=runs/verification-csd
python -m scripts.acquire_csd data/cleared/csd/CSD.zip "$DATA"
python -m scripts.prepare_lyrics freeze "$DATA/sources.jsonl" "$DATA/rights.jsonl" \
  "$OUT/prepared" --seed 0 --max-seconds 32
python -m scripts.prepare_lyrics verify "$OUT/prepared"
/usr/bin/time -p python -m chorustrace train "$OUT/prepared/manifest.jsonl" "$OUT/baseline.npz" \
  --split train --condition vocal --architecture baseline --epochs 30 --hop-size 320 --seed 0 \
  2> "$OUT/train.log"
python -m chorustrace evaluate "$OUT/baseline.npz" "$OUT/prepared/manifest.jsonl" \
  --split train --samples 0 > "$OUT/train-score.json"
```

## Protocol

1. Review recording, lyric, training, and checkpoint permissions. Verify time boundaries and performer, recording, and canonical song identifiers. See [data rights](../docs/data-rights.md) and [data preparation](../docs/data-preparation.md).
2. Freeze the split and hash its manifest and rendered WAVs before fitting. Artists, recordings, and song versions remain in a single split. Only available audio conditions are rendered.
3. Compare baseline and longer-context models with the same isolated training clips and validation artists. Default settings are 30 epochs, width 48, learning rate 0.003, 320-sample feature hop, and seed 0. The models see up to 27 and 81 frames respectively. Select by vocal validation WER, with CER as a tiebreaker.
4. Where paired validation instrumentals exist, fit a gate using validation only. A clip's confidence is the mean of the peak posterior for each emitted non-space CTC character. The chosen threshold is the smallest that makes the validation instrumental false-alarm rate strictly less than 5%. Count rejected voiced clips as deletions.
5. Load the selected checkpoint in a new process and score the held-out test split once. WER and CER are micro-averaged across voiced clips. CER includes spaces. Instrumental false alarms use the number of instrumental clips as the denominator. The JSON report gives edit counts, denominators, sample predictions, and a condition breakdown for each genre.

## Synthetic implementation checks

The original one-clip smoke fit and the version-1 fixture checkpoint each decoded `hello moon` after reload. The new artist-disjoint synthetic fixture has 16 training excerpts (22.1899375 s), eight validation excerpts (10.9105 s), and nine test excerpts (12.857875 s). Its only genre is `synthetic`. The learning-curve script fit the available training subset because the fixture has under an hour of singing.

| Model | Steps | Fit time | Training WER / CER | Validation WER / CER |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 480 | 0.42 s | 26.00% / 7.11% | 104.17% / 60.40% |
| Long context | 480 | 0.52 s | 6.00% / 1.42% | 116.67% / 65.35% |

The baseline won on validation. Its threshold of `0.39175794614585163` reduced validation instrumental false alarms from 8/8 to 0/8 and suppressed 0/8 vocal and 2/8 original-mix drafts. On the test split every voiced condition had 29 reference words and 126 reference characters across nine clips.

| Synthetic test condition | Clips | Greedy WER / CER | Gated WER / CER | Voiced drafts suppressed |
| --- | ---: | ---: | ---: | ---: |
| Isolated vocal | 9 | 124.14% / 64.29% | 124.14% / 64.29% | 0 |
| Original procedural mix | 9 | 113.79% / 75.40% | 110.34% / 79.37% | 3 |
| Accompaniment −12 dB | 9 | 120.69% / 76.19% | 117.24% / 84.13% | 4 |
| Accompaniment −6 dB | 9 | 117.24% / 73.02% | 113.79% / 78.57% | 3 |
| Accompaniment 0 dB | 9 | 117.24% / 76.19% | 113.79% / 80.16% | 3 |
| Accompaniment +6 dB | 9 | 100.00% / 74.60% | 100.00% / 82.54% | 3 |
| Instrumental | 9 | 9/9 false alarms | 0/9 false alarms | — |

The fixture tests alignment, conditions, calibration, and artifact reloads. The synthetic false-alarm count measures nine procedural backings.

### Marked chorus diagnostic

A separate seed-29 three-clip fit memorized the procedural chorus variants. The independent decoder recovered the changed word `today`. Pooling dropped it from the changed take. This is an audio-driven diagnostic of the pooling behavior on the fitted clips. The test-split scores above use independent drafts.

### Reproduce synthetic runs

Run from the repository root using Python 3.10 or later and NumPy. Fit times in this report are local wall times.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m chorustrace prepare-synthetic runs/verification-smoke
python -m chorustrace train runs/verification-smoke/smoke.jsonl \
  runs/verification-smoke/smoke.npz --epochs 80 --learning-rate 0.008 --hop-size 320 --seed 0
SMOKE_WAV=$(python -c 'import json; from pathlib import Path; p=Path("runs/verification-smoke/smoke.jsonl"); print(p.parent / json.loads(p.read_text())["audio"])')
python -m chorustrace transcribe runs/verification-smoke/smoke.npz "$SMOKE_WAV"
python -m chorustrace transcribe benchmarks/checkpoints/synthetic_smoke.npz "$SMOKE_WAV"

python -m scripts.make_experiment_fixture runs/verification-synthetic
python -m scripts.prepare_lyrics verify runs/verification-synthetic/prepared
python -m scripts.learning_curve runs/verification-synthetic/prepared \
  runs/verification-synthetic/curve-baseline --architecture baseline
python -m scripts.learning_curve runs/verification-synthetic/prepared \
  runs/verification-synthetic/curve-long-context --architecture long-context
python -m scripts.abstention_experiment runs/verification-synthetic/curve-baseline/available.npz \
  runs/verification-synthetic/prepared/manifest.jsonl runs/verification-synthetic/abstention.json --fixture
python -m scripts.benchmark_lyrics runs/verification-synthetic/prepared \
  runs/verification-synthetic/curve-baseline/available.npz --split test \
  --calibration runs/verification-synthetic/abstention.json > runs/verification-synthetic/test.json
```

```sh
python -m chorustrace prepare-synthetic runs/verification-chorus --seed 29
python - <<'PY'
from pathlib import Path
from chorustrace.data import read_jsonl, write_jsonl

root = Path('runs/verification-chorus')
rows = [dict(row, split='train') for _, row in read_jsonl(root / 'manifest.jsonl')
        if row['song_id'] == 'test-01' and row['clip_id'] in {'repeat-1', 'repeat-2', 'changed'}
        and row['condition'] == 'vocal']
write_jsonl(root / 'diagnostic.jsonl', rows)
PY
python -m chorustrace train runs/verification-chorus/diagnostic.jsonl \
  runs/verification-chorus/diagnostic.npz --epochs 140 --learning-rate 0.008 --hop-size 320 --seed 0
python -m chorustrace chorus runs/verification-chorus/diagnostic.npz \
  runs/verification-chorus/diagnostic.jsonl runs/verification-chorus/repeats.jsonl \
  --split train --condition vocal > runs/verification-chorus/diagnostic.json
```
