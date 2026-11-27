# Full-track evaluation

## Measured results

The real holdout is frozen at SHA-256 `9fa1241d905db1d31b1f1695303bb1cecb8c59f1ed1d93aad15cb1fb62ec22c2`. It has **20 complete English songs** and five auxiliary instrumental recordings. See the frozen-release protocol below. The earlier development measurements use generated material and historical excerpts.

| Frozen 20-song result | Words | S | D | I | WER | CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock checkpoint before real training | 5,780 | 1,317 | 4,415 | 52 | 100.07% | 89.00% |
| Validation-selected checkpoint, fresh process | 5,780 | 30 | 5,726 | 0 | 99.58% | 99.65% |

The 2.0% goal remains unmet. The selected checkpoint emitted eight false words in 22.24 minutes of auxiliary instrumentals, or 0.36 words/minute, versus 2,269 false words for the stock model. Each of the five instrumental tracks emitted at least one word. The segmented and single-pass transcripts were identical for all 20 lyrical mixes over 209 joins. Word-timed near-join attribution is unavailable on this release. A second fresh process reproduced the whole report byte for byte. [Per-track scores, CER, diagnostics and sample predictions](results/clip1_frozen_scores.json)

The checkpoint selected using only the same 35 development excerpts uses the default acoustic dilation, epoch 9, 276 groups and a blank offset of zero. Within the artist-disjoint expanded run, the best validation WER at 736, 1,406 and 2,766 training excerpts was 95.28%, 94.88% and 95.28% respectively. The usable training pools contained 433 excerpts and 2,766 excerpts, totaling 33.06 minutes and 3.45 hours. An independent audit discovered alias-level validation overlap during early fits. Those fits were discarded, collaborator identities were added, and all reported candidates were trained again from random initialization. The expanded run is scoped to local noncommercial research.

The selected checkpoint from the first experiment is at `runs/clip1-development/selected.npz` and its SHA-256 is `34da3e6b413358e9ae9c4dde22a5c33c1c9f96851016370a8eb70ea76535ee5e`. The complete validation sweep, pre-goal baseline and fresh test reports are in the same directory. The remaining accuracy gap is dominated by deletions. The real training pool has 3.45 hours of phrase excerpts. The follow-up below expands validation while retaining all 35 original validation clips.

The original implementation at `6a3132d` processed each available historical original-mix prefix as one continuous input, then scored the known intervals. Its twenty-song checkpoint scored **98.61% WER and 81.91% CER** on 432 words across six recordings. Its five-song checkpoint scored **104.63% WER and 87.31% CER** on the same intervals. On three continuous assembled synthetic mixes, its public synthetic checkpoint scored **100.00% WER and 62.88% CER** on all 29 words. The report contains per-input results and prediction previews. [Legacy probe](results/legacy_full_input.json)

The new checkpoint was initialized at random and selected on two generated full mixes. Each scale used a nested, artist-grouped subset. Validation ran every five epochs. Selection minimized aggregate WER, then CER, then preferred the smaller training scale and earlier epoch on ties.

| Training groups | Training seconds | Selected epoch | Validation WER | Validation CER |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 68.21 | 90 | 97.50% | 79.89% |
| 4 | 137.65 | 80 | 97.50% | 64.67% |
| 8 | 273.90 | 85 | 92.50% | 54.35% |

The saved eight-group checkpoint reproduced these scores in a fresh process. Five of 12 instrumental intervals contained an emitted word. Two edits fell within 0.5 seconds of the two segment joins. The matched single-pass control found zero additional near-join edits from segmentation. The reload report includes each track's prediction. [Saved checkpoint](checkpoints/synthetic_fulltrack.npz) · [Learning curve](results/synthetic_fulltrack_training.json) · [Fresh reload with predictions](results/synthetic_fulltrack_reload.json)

| Generated validation track | Words | S | D | I | WER | CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generated-validation-01 | 20 | 12 | 7 | 0 | 95.00% | 57.00% |
| generated-validation-02 | 20 | 13 | 5 | 0 | 90.00% | 51.19% |
| Aggregate | 40 | 25 | 12 | 0 | 92.50% | 54.35% |

### Reproduce measured results

Run from the repository root with Python 3.12 and NumPy installed. The legacy diagnostic requires the two historical local checkpoints and the audio prefixes and manifest from the [MUSDB18 experiment](musdb18_genre20.md).

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p runs
legacy_tree=$(mktemp -d)
git archive 6a3132d chorustrace | tar -x -C "$legacy_tree"
PYTHONPATH="$legacy_tree" .venv/bin/python -P benchmarks/score_legacy_full_input.py \
  > runs/legacy_full_input.json

PYTHONPATH=. .venv/bin/python benchmarks/train_synthetic_fulltrack.py \
  --epochs 100 --validate-every 5 --checkpoint runs/synthetic-fulltrack.npz \
  --report runs/synthetic-fulltrack-training.json
PYTHONPATH=. .venv/bin/python benchmarks/train_synthetic_fulltrack.py \
  --reload --checkpoint runs/synthetic-fulltrack.npz \
  --report runs/synthetic-fulltrack-reload.json
```

## Dataset and freeze

Keep a development JSONL and a separate test JSONL. Resolve asset paths relative to their manifest. Each row gives a unique `track_id`, canonical `song_id`, every credited `artist_ids` entry, `lyric_id` for a shared lyric work, and `recording_family_id` for versions of one recording. Reuse identities across aliases, alternate takes, covers and clips as applicable. `audio_source`, `recording_permission`, and `lyrics_permission` point to evidence for the exact recording and lyric work and the declared use. Confirm `source_frames` and `source_sample_rate` against the source before accepting a WAV conversion.

```json
{"track_id":"mix-01","song_id":"work-01","artist_ids":["artist-01"],"lyric_id":"lyric-01","recording_family_id":"recording-01","genre":"hip-hop","split":"test","audio":"audio/mix-01.wav","reference":"references/mix-01.json","condition":"original_mix","language":"en","full_track":true,"audio_source":"release evidence","source_frames":7938000,"source_sample_rate":44100,"recording_permission":"recording license evidence","lyrics_permission":"lyric license evidence"}
```

Each reference file contains `text`, `complete_lead_vocal: true`, and `transcribed_by`. Test references also require a different `checked_by` and a `review_evidence` pointer. Preserve the complete order of audible lead lyrics, including spoken or rapped lead lines, audible lexical ad-libs such as “oh,” every repeated word, and every changed repetition. Exclude backing-only lyrics from the lead transcript. For timing diagnostics, supply `words` with increasing, nonoverlapping times in seconds. Annotate verified instrumental and silent passages in `instrumental_intervals` as `[start, end]` pairs. Leave passages with distinct backing vocals out of this inventory. Use `[]` for a fully checked track with no instrumental passage and omit the field when instrumental annotation is unavailable. The test corpus requires at least one annotated passage and the report gives annotated coverage. A train row may use `full_track: false` with `condition: "vocal"` or `"original_mix"` and a complete untimed label for the supplied clip. Longer training mixes require timed words to produce aligned clips.

The freeze requires 20 distinct works and recording families, spanning pop, rock, hip-hop, and at least one other genre. It records WAV, PCM, reference, and normalized lyric fingerprints. Keep a trusted copy of the lock and supporting records with the independent evaluator. The checks rely on accurate identities, provenance and reviewer attestations. Training checks the lock without opening the test files, and the saved checkpoint records the lock hash. Use the fixed decoder and metrics for the selected checkpoint. Preserve the reports from the independent run.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p runs/fulltrack
.venv/bin/python -m chorustrace freeze-test data/test/manifest.jsonl runs/fulltrack/test.lock.json
.venv/bin/python -m chorustrace verify-test data/test/manifest.jsonl runs/fulltrack/test.lock.json
.venv/bin/python -m chorustrace train-full data/development/manifest.jsonl \
  runs/fulltrack/test.lock.json runs/fulltrack/selected.npz --epochs 30 --seed 0 \
  > runs/fulltrack/learning-curve.json
.venv/bin/python -m chorustrace transcribe runs/fulltrack/selected.npz path/to/full-mix.wav
.venv/bin/python -m chorustrace evaluate-full runs/fulltrack/selected.npz \
  data/test/manifest.jsonl runs/fulltrack/test.lock.json --samples 3 \
  > runs/fulltrack/test-results.json
```

The decoder uses 20-second segments with the acoustic model's full context at every join. It collapses CTC across the resulting frame sequence once and produces timed words and one transcript. The evaluator sums word substitutions, deletions and insertions and divides by the total normalized reference words. CER includes spaces. Instrumental false alarms use word midpoints inside annotated intervals. Paired evaluation compares segmented and single-pass lexical edits, including tracks without reference word timestamps. Reported near-join attribution applies to timed references, with its track and join coverage listed separately.

`train-full` keeps log-mel features in a temporary local file and loads one clip at a time. At default feature settings, allow about 115 MB of temporary feature data per audio hour plus space for the prepared training clips. On the same 72-minute generated input with one epoch and a hidden size of four, peak process memory fell from 352 MiB to 56 MiB. The generated experiment's saved checkpoint and reports reproduced byte for byte after the change.

## Public data provenance and staging

- Staged 10 Sean McGrath recordings with author charts under `data/public_training/sean_mcgrath/`. The WAVs total 38.16 minutes. McGrath licenses both the original recordings and song materials under CC BY 4.0. Credit Sean McGrath, `@seanmcgrathmusic`. Each chart still needs to be checked against the performed rendition and aligned. Reserve this artist for training. [Author and lyric rights](https://seanmcgrathmusic.com/) · [Audio and recording rights](https://seanmcgrathmusic.com/mp3s/)
- Staged 17 English vocadito clips from 15 anonymized performers, totaling 6.00 minutes, under `data/public_training/vocadito/english/`. The dataset authors collected permissions and used original or public-domain compositions. Reserve these isolated vocals for training development and reconcile the work and performer identities against the final test manifest before using them. Use `--clip-seconds 32` to admit the longest clip. [CC BY 4.0 dataset](https://zenodo.org/records/5578807) · [Recording and composition provenance](https://archives.ismir.net/ismir2021/latebreaking/000044.pdf)
- Staged seven full hip-hop candidate WAVs under `data/public_test_candidates/holizna/`. The total is 24.62 minutes. The inventory records release links, frame counts, and file hashes. Attribute Holizna under the release's CC BY 4.0 license. All seven need independently checked lead-vocal references before the freeze. Other release sources for candidate review include Lorenzo's Music's original art-rock EP and Ryan Cornett's original hymns. [Holizna artist release and license](https://holiznacc0.bandcamp.com/album/good-kids-mixtape-1) · [Artist WAV downloads](https://www.patreon.com/Holizna/posts/pastries-behind-68123911) · [Lorenzo's Music](https://faircamp.lorenzosmusic.com/spaghetti-mid-western/index.html) · [Ryan Cornett](https://ryancornett.com/)
- The 20 complete English Jam-ALT v1.4.0 WAVs total 72.22 minutes. The source MP3 SHA-256 and file size matched between the independently reviewed release and the explicitly per-track licensed JamendoLyrics release for all 20. Use the reviewed references for local benchmarking under the source terms and the authors' published evaluation instructions. Retain the lead by removing marked backing vocals, preserve repeated occurrences, and normalize the one spoken number using its exact match in the original human-timed word entries. The older timings apply to a different wording revision and the frozen lead references use the reviewed lexical text without borrowed timings. [Jam-ALT evaluation instructions](https://audioshake.github.io/jam-alt/) · [Independent review and annotation policy](https://arxiv.org/html/2408.06370v1#S2) · [Pinned original license](https://huggingface.co/datasets/jamendolyrics/jamendolyrics/blob/e9a8a63e7c828a0a7152c46752de5c24ae65de94/LICENSE)
- The same holdout lock includes five RWC Jazz instrumental recordings totaling 22.24 minutes. Decode each full original WAV and count predicted words within its annotated music-only span. The report separates this diagnostic from the 20-song WER and CER. The full-song WER still counts any extra words emitted in instrumental passages. [RWC release](https://zenodo.org/records/18656623) · [Official instrumentation metadata](https://github.com/rwc-music/rwc-annotations/blob/main/metadata.csv)
- Staged 24 RWC 2.0 original English mixes totaling 96.05 minutes under `data/public_test_candidates/rwc/`, with review packets under `data/reference_review/rwc/`. The release uses production masters under CC BY-NC 4.0 for noncommercial research. The selection covers 10 credited artists, eight pop songs, six rock and metal, three hip-hop, and seven from other genres. All 24 have published lyric drafts. Eight have separate activity annotations to assist the review. Keep RWC performers, songs, and alternate recordings outside training and validation. [Audio and license](https://zenodo.org/records/18656623) · [Release paper](https://doi.org/10.5334/tismir.326) · [Metadata](https://github.com/rwc-music/rwc-annotations/blob/main/metadata.csv) · [Published drafts](https://github.com/rwc-music/rwc-annotations-archive)
- Additional review candidates include four full Lorenzo's Music releases under CC BY-SA 4.0 and nine mixes from the original 2008 Brad Sucks download under CC BY-SA 3.0. These WAVs preserve the released mix and were decoded from the artists' public Opus and MP3 copies. The Brad Sucks candidate selection omits the track with extra community vocal credits. [Lorenzo's Music release](https://faircamp.lorenzosmusic.com/spaghetti-mid-western/index.html) · [Brad Sucks release](https://www.bradsucks.net/news/archives/2008/09/08/out-of-it) · [Brad Sucks license record](https://wiki.creativecommons.org/wiki/Case_Studies/Brad_Sucks)
- The CLIP1 training catalog has 2,905 unprocessed English lyric excerpts with unambiguous text. Of those, 449 carry catalog labels for CC BY, CC BY-SA, or CC0. Cross-checking those labels and artist identities against the FMA catalog and current reservations left 440 candidates from 79 catalog artists and 162 source tracks in `data/public_training/clip1_audit/shortlist.jsonl`. Five original-mix demo excerpts totaling 17.38 seconds were decoded to WAV and validated with their exact published annotations under `data/public_training/clip1_audit/demo-training/`. Four demo excerpts carry noncommercial terms. The release README points to each original track's terms. Match canonical artist, song and lyric identities against the final reservations before training. CLIP1 labels apply to the cropped excerpts, with boundaries checked against the music by the dataset authors. [CLIP1 release](https://zenodo.org/records/17950664) · [Annotation method](https://pmc.ncbi.nlm.nih.gov/articles/PMC12861282/) · [FMA upload policy](https://freemusicarchive.org/Upload_Policy)

```sh
mkdir -p data/public_training/vocadito
curl -fL 'https://zenodo.org/records/5578807/files/vocadito.zip?download=1' \
  -o data/public_training/vocadito/vocadito.zip
PYTHONPATH=. .venv/bin/python benchmarks/prepare_vocadito.py \
  data/public_training/vocadito/vocadito.zip data/public_training/vocadito/english
```

### Prepare the independent review

The following commands select the RWC waveforms from the release ZIPs using byte ranges. They verify the compressed entries and the full WAV durations. The review helper checks the PCM inventory and writes two private, blank listening passes against the same WAV hashes. Published lyric drafts and activity labels can support reconciliation after listening. Freeze only after at least 20 tracks across the target genres have independently checked lead-vocal references.

```sh
mkdir -p data/public_test_candidates/rwc data/reference_review
curl -fLsS 'https://raw.githubusercontent.com/rwc-music/rwc-annotations/main/metadata.csv' \
  -o data/public_test_candidates/rwc/metadata.csv
.venv/bin/python benchmarks/prepare_rwc_test_candidates.py \
  data/public_test_candidates/rwc/metadata.csv data/public_test_candidates/rwc
PYTHONPATH=. .venv/bin/python benchmarks/prepare_reference_review.py \
  data/public_test_candidates/rwc/inventory.json data/reference_review/rwc
```

### Reproduce the frozen release experiment

Use the pinned-source acquisition helper to restore the original frozen manifest and lock in a new directory. It checks source and PCM hashes and uses FFmpeg 8.1.1 for the source MP3-to-WAV conversion. The original track and lyric terms apply. Build the local checkpoint using the training commands below. Evaluation and transcription use Python and NumPy offline.

```sh
.venv/bin/python -m benchmarks.acquire_frozen_fulltrack local-replay/data
replay=local-replay/data/frozen_fulltrack
checkpoint=runs/clip1-acoustic-v2/selected.npz
.venv/bin/python -m chorustrace evaluate-full "$checkpoint" \
  "$replay/manifest.jsonl" "$replay/test.lock.json" --samples 3
.venv/bin/python -m chorustrace transcribe "$checkpoint" /path/to/full-track.wav
```

Use the staged inventories, original CLIP1 archives, and fixed validation shortlist in `data/clip1_development/audit/` to rebuild the training data. An existing frozen directory can go directly to `verify-test`. Skip completed staging commands when their outputs already exist.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=.
mkdir -p runs/clip1-development
.venv/bin/python -m chorustrace verify-test \
  data/frozen_fulltrack/manifest.jsonl data/frozen_fulltrack/test.lock.json
.venv/bin/python benchmarks/score_frozen_baseline.py \
  data/frozen_fulltrack/manifest.jsonl data/frozen_fulltrack/test.lock.json \
  > runs/clip1-development/stock-fulltrack-baseline.json

.venv/bin/python benchmarks/screen_clip1_development.py \
  data/public_training/clip1_audit data/frozen_fulltrack/manifest.jsonl \
  data/frozen_fulltrack/test.lock.json data/clip1_development/audit-disjoint \
  --baseline data/clip1_development/audit
.venv/bin/python benchmarks/prepare_clip1_training.py \
  data/clip1_development/audit-disjoint data/public_training/clip1_audit/cadenza_clip1_data.train.v1.2.tar.gz \
  data/clip1_development/training --max-clip-seconds 15
.venv/bin/python benchmarks/prepare_clip1_training.py \
  data/clip1_development/audit-disjoint data/public_training/clip1_audit/cadenza_clip1_data.valid.v1.2.tar.gz \
  data/clip1_development/validation --split validation
```

The screen excludes reserved performer and work identities and exact occurrences of training excerpt text or shared six-word phrases inside the test references. The combined development manifest carries individual collaborator identities, and the trainer checks them against validation and the test lock. CLIP1 validation uses 35 independently released original-mix excerpts from nine works and eight catalog artist IDs. It has no rap excerpts. The frozen WER run uses 20 complete songs.

```sh
.venv/bin/python benchmarks/combine_clip1_development.py \
  data/clip1_development/audit-disjoint data/clip1_development/training/train.jsonl \
  data/clip1_development/validation/validation.jsonl data/clip1_development/disjoint-manifest.jsonl

for model in default wider; do
  if [ "$model" = default ]; then
    set -- 1 3 9
  else
    set -- 1 3 9 27 81
  fi
  .venv/bin/python -m chorustrace train-full data/clip1_development/disjoint-manifest.jsonl \
    data/frozen_fulltrack/test.lock.json "runs/clip1-development/$model-disjoint.npz" \
    --epochs 18 --validate-every 3 --clip-seconds 15 --dilations "$@" \
    > "runs/clip1-development/$model-disjoint-training.json"
done

.venv/bin/python benchmarks/screen_clip1_development.py \
  data/public_training/clip1_audit data/frozen_fulltrack/manifest.jsonl \
  data/frozen_fulltrack/test.lock.json data/clip1_development_expanded/audit-disjoint \
  --baseline data/clip1_development/audit --allow-noncommercial
.venv/bin/python benchmarks/prepare_clip1_training.py \
  data/clip1_development_expanded/audit-disjoint \
  data/public_training/clip1_audit/cadenza_clip1_data.train.v1.2.tar.gz \
  data/clip1_development_expanded/training --max-clip-seconds 30 --allow-noncommercial

.venv/bin/python benchmarks/combine_clip1_development.py \
  data/clip1_development_expanded/audit-disjoint data/clip1_development_expanded/training/train.jsonl \
  data/clip1_development/validation/validation.jsonl data/clip1_development_expanded/disjoint-manifest.jsonl

.venv/bin/python -m chorustrace train-full data/clip1_development_expanded/disjoint-manifest.jsonl \
  data/frozen_fulltrack/test.lock.json runs/clip1-development/expanded.npz \
  --epochs 12 --validate-every 3 --clip-seconds 30 \
  > runs/clip1-development/expanded-training.json
.venv/bin/python benchmarks/select_clip1_checkpoint.py \
  data/clip1_development/disjoint-manifest.jsonl data/frozen_fulltrack/test.lock.json \
  runs/clip1-development/selected.npz runs/clip1-development/default-disjoint.npz \
  runs/clip1-development/wider-disjoint.npz runs/clip1-development/expanded.npz \
  > runs/clip1-development/selection.json
.venv/bin/python -m chorustrace evaluate-full runs/clip1-development/selected.npz \
  data/frozen_fulltrack/manifest.jsonl data/frozen_fulltrack/test.lock.json --samples 3 \
  > runs/clip1-development/test-results.json
```

The expanded experiment includes source tracks eligible for local noncommercial research. Keep their audio, lyric files and derived local artifacts under the applicable track terms. The calibration script tests offsets `0, 0.5, 1, 1.5, 2, 3, 4` on validation. It selects on WER, then CER, then the smaller offset. It preserves the first supplied checkpoint on an exact tie.

## Expanded validation and CTC follow-up

The expanded validation manifest preserves the 2,766 audited music training clips and all 35 continuity clips. It adds 312 official CLIP1 validation excerpts across licensed conditions. The full validation split has 347 excerpts, 2,418 reference words and nine hip-hop clips. A separate, performer-disjoint RWC Pop development bank covers 135.64 seconds in 34 human-annotated instrumental-solo intervals. The bank is used as a diagnostic after lexical checkpoint selection. The test lock remains the same.

The acoustic ablation compares 160-, 320- and 480-sample hops in a fixed 16 kHz feature pipeline with 276 training groups and 1,406 music excerpts per run. The 480-sample hop uses a 512-sample window, and both shorter hops use 400 samples. The 480-sample setting lowers the training frames per minimum CTC path from 13.17 to 4.41 while retaining a feasible alignment for every music training clip. The delta front end learns from centered log-mel slopes. The prior arm reweights CTC paths during training using previous-epoch token probabilities estimated on training audio. Its inference uses the learned acoustic parameters only. [CTC label-prior derivation](https://arxiv.org/html/2406.02560v3#S2.SS3)

The same 320-sample music arm also tests 286 Libri-Light spoken excerpts from 24 readers with their released orthographic transcriptions. It trains jointly from random weights on the fixed 1,406 music excerpts and 62.44 minutes of CC BY 4.0 speech. Both arms have 18 epochs, giving the joint arm more updates per epoch. [LibriSpeech grant](https://www.openslr.org/12/) · [Libri-Light subset](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md#limited-supervision-dataset)

The beam marginalizes CTC blank and nonblank paths across the complete track and uses a Viterbi path to time the chosen prefix. Decoding uses acoustic scores and the CTC state. The comparison ranks greedy candidates on validation WER and CER, then tests beam widths four and eight and fixed blank offsets on the top two. One checkpoint contains all decoder settings needed to reload it in a separate process.

On 2,418 validation words, the 20 ms hop with greedy decoding scored 96.36% WER and 94.50% CER. It made four fewer word edits than the older 10 ms checkpoint on the same excerpts. Beam decoding and blank penalties reduced deletions in several runs while increasing other edits. The delta, wider-encoder, prior and joint speech candidates also scored higher validation WER. The music-only winner generated zero words inside the 34 annotated development instrumental spans, as did the incumbent.

| Same frozen 20-song suite | S | D | I | WER | CER | False words in 22.24 instrumental minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First validation-selected checkpoint | 30 | 5,726 | 0 | 99.58% | 99.65% | 8 |
| Follow-up validation-selected checkpoint | 122 | 5,631 | 0 | 99.53% | 98.82% | 88 |

The follow-up recovered 27 of 5,780 words correctly, an increase of three. It also emitted 80 more words in the separate frozen instrumentals, for 3.96 false words per minute. Segmented and single-pass transcripts matched on every song across 209 joins. Gold near-join word times are unavailable. Two separate processes produced identical output reports. This was a re-evaluation on the same frozen cohort, with model and decoder selection confined to validation. The 2.0% target remains unmet. [Validation sweep, 20 track scores and predictions](results/clip1_ctc_followup.json)

The follow-up checkpoint is `runs/clip1-acoustic-v2/selected.npz` with SHA-256 `6c520f8df70407fed65e01506c3bc05e519e966b17775485c3cec5823e834ff3`. The next data input is a larger collection of rights-cleared original mixes with aligned lead vocals, including full-song or longer-span development recordings across the target genres. The speech ablation adds an hour of phonetic supervision and shows little improvement in lyric WER at the measured scale.

### Reproduce the follow-up

Run from the repository root with the original verified CLIP1 archives, the original frozen lock and the staged split audit. Skip staging steps whose verified destination already exists.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=.
dev=data/clip1_validation_expanded/disjoint-manifest.jsonl
test=data/frozen_fulltrack/manifest.jsonl
lock=data/frozen_fulltrack/test.lock.json
run=runs/clip1-acoustic-v2
mkdir -p "$run"

.venv/bin/python -m benchmarks.screen_clip1_validation data/public_training/clip1_audit \
  data/clip1_development_expanded/audit-disjoint "$test" "$lock" data/clip1_validation_expanded/audit
.venv/bin/python -m benchmarks.prepare_clip1_training data/clip1_validation_expanded/audit \
  data/public_training/clip1_audit/cadenza_clip1_data.valid.v1.2.tar.gz \
  data/clip1_validation_expanded/validation --split validation --allow-noncommercial --max-clip-seconds 30
.venv/bin/python -m benchmarks.combine_clip1_development data/clip1_validation_expanded/audit \
  data/clip1_development_expanded/training/train.jsonl \
  data/clip1_validation_expanded/validation/validation.jsonl "$dev"
.venv/bin/python -m benchmarks.prepare_rwc_instrumental_development data/public_test_candidates/rwc \
  "$dev" "$lock" data/rwc_instrumental_development

fit() {
  name=$1
  shift
  .venv/bin/python -m chorustrace train-full "$dev" "$lock" "$run/$name.npz" \
    --epochs 18 --validate-every 3 --clip-seconds 30 --scale-groups 276 "$@" > "$run/$name.json"
}
fit hop-160 --hop-size 160
fit hop-320 --hop-size 320
fit hop-480 --hop-size 480 --window-size 512
fit hop-320-delta --hop-size 320 --input-deltas
fit hop-480-delta --hop-size 480 --window-size 512 --input-deltas
fit hop-320-prior --hop-size 320 --ctc-prior-weight 0.3
fit hop-320-width96 --hop-size 320 --hidden-size 96

speech=data/public_training/libri_light
mkdir -p "$speech"
curl -fLsS 'https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz' \
  -o "$speech/librispeech_finetuning.tgz"
.venv/bin/python -m benchmarks.prepare_libri_light_training "$speech/librispeech_finetuning.tgz" \
  data/clip1_validation_expanded/validation/validation.jsonl "$test" "$lock" "$speech/one_hour"
.venv/bin/python -m benchmarks.combine_auxiliary_speech "$dev" "$speech/one_hour/train.jsonl" \
  "$lock" "$run/song-and-speech.jsonl" --music-groups 276
.venv/bin/python -m chorustrace train-full "$run/song-and-speech.jsonl" "$lock" "$run/hop-320-speech.npz" \
  --epochs 18 --validate-every 3 --clip-seconds 30 --scale-groups 300 --hop-size 320 \
  > "$run/hop-320-speech.json"

.venv/bin/python -m benchmarks.compare_ctc_validation "$dev" "$lock" "$run/selected.npz" \
  runs/clip1-development/selected.npz "$run/hop-160.npz" "$run/hop-320.npz" "$run/hop-480.npz" \
  "$run/hop-320-delta.npz" "$run/hop-480-delta.npz" "$run/hop-320-speech.npz" "$run/hop-320-prior.npz" \
  "$run/hop-320-width96.npz" \
  --instrumental data/rwc_instrumental_development/manifest.jsonl > "$run/selection.json"
.venv/bin/python -m chorustrace evaluate-full "$run/selected.npz" "$test" "$lock" --samples 3 \
  > "$run/test-results.json"
```

## Training scale, stereo features, and instrumental exposure

All runs started from random weights and used the same 347 CLIP1 validation excerpts with 2,418 reference words. The scale comparison used nested music groups and the same 20 ms frame hop. The phoneme arm used the published clip-level targets as auxiliary supervision only during training. The stereo arm added a clipped log-mel side-to-mid power ratio to the mono log-mel features. Mono inputs supplied zeros for the side-to-mid feature. The instrumental arm kept the exact 1,406-clip music roster and added 23 blank-only excerpts totaling 70.53 seconds. It reserved 14 manually annotated solo spans totaling 65.11 seconds, disjoint by performer group.

| Training condition | Music groups | Music clips | Selected epoch | Validation WER | Validation CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 ms mono, smaller subset | 138 | 736 | 6 | 96.77% | 95.08% |
| 20 ms mono, incumbent | 276 | 1,406 | 9 | 96.36% | 94.50% |
| 20 ms mono, all available clips | 552 | 2,766 | 6 | 96.65% | 97.17% |
| Auxiliary phoneme CTC | 276 | 1,406 | 3 | 96.40% | 95.78% |
| Stereo features | 276 | 1,406 | 9 | 96.86% | 95.59% |
| Blank-only instrumental exposure | 276 | 1,406 | 12 | 96.86% | 94.50% |

The validation sweep compared 17 checkpoint and decoder combinations. The incumbent again had the lowest validation WER and emitted zero words on the reserved solo spans. The local selected checkpoint has SHA-256 `6c520f8df70407fed65e01506c3bc05e519e966b17775485c3cec5823e834ff3`. A fresh process loaded it and reproduced the existing frozen-test report byte for byte. The 20 tracks still score **99.53% WER** with 122 substitutions, 5,631 deletions, and zero insertions across 5,780 lead-vocal words. The complete score set includes 98.82% CER and 88 words on the five frozen instrumental diagnostics. [Validation curves and decoder sweep](results/clip1_training_followup.json) · [20 per-track scores and sample predictions](results/clip1_ctc_followup.json)

From the repository root, use the staged CLIP1 data, RWC Pop bank, and frozen test described above. The staging destinations must be new directories.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=.
dev=data/clip1_validation_expanded/disjoint-manifest.jsonl
test=data/frozen_fulltrack/manifest.jsonl
lock=data/frozen_fulltrack/test.lock.json
run=runs/clip1-acoustic-repeat
phones=data/clip1_validation_expanded/phoneme_targets_repeat
split=data/rwc_instrumental_split_repeat
mkdir -p "$run"

.venv/bin/python -m benchmarks.prepare_clip1_phoneme_targets "$dev" "$lock" \
  data/public_training/clip1_audit/cadenza_clip1_data.metadata.v1.3.tar.gz "$phones"
.venv/bin/python -m benchmarks.prepare_instrumental_training \
  data/rwc_instrumental_development/manifest.jsonl "$dev" "$lock" "$split"

fit() {
  name=$1
  manifest=$2
  shift 2
  .venv/bin/python -m chorustrace train-full "$manifest" "$lock" "$run/$name.npz" \
    --epochs 18 --validate-every 3 --clip-seconds 30 --hop-size 320 "$@" > "$run/$name.json"
}
fit phoneme "$dev" --scale-groups 276 --phoneme-supervision "$phones/phoneme-targets.json" \
  --phoneme-loss-weight 0.3
fit scale-20ms "$dev" --scale-groups 138 552
fit stereo "$dev" --scale-groups 276 --stereo-features
fit instrumental "$split/combined.jsonl" --scale-groups 278

.venv/bin/python -m benchmarks.compare_ctc_validation "$split/combined.jsonl" "$lock" \
  "$run/selected.npz" runs/clip1-acoustic-v2/selected.npz "$run/phoneme.npz" \
  "$run/scale-20ms.npz" "$run/stereo.npz" "$run/instrumental.npz" \
  --instrumental "$split/validation-diagnostics.jsonl" --joint-instrumental > "$run/selection.json"
.venv/bin/python -m chorustrace evaluate-full "$run/selected.npz" "$test" "$lock" \
  --samples 3 > "$run/test-results.json"
```

The verified CLIP1 training pool contains 3.45 hours of original-mix excerpts. The 20 ms learning curve reached its lowest WER at 1.73 hours. Continued scaling needs a larger English corpus with original mixes, timestamped performed lead lyrics, permissions for both recordings and lyrics, and artist, work, and alternate-version identifiers for holdout screening. The RWC catalog lists about four hours of English tracks, and its published lyric drafts still need timing. SourceAudio advertises timed lyrics and original mixes under an opt-in licensing program that requires an agreement. [RWC release](https://zenodo.org/records/18656623) · [SourceAudio licensing](https://www.sourceaudio.com/ai-dataset-licensing/)
