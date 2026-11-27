# Prepare and freeze a lyric corpus

Keep rights evidence, source WAVs, and timed annotations in a local data directory. Convert audio to integer PCM WAV before preparation. Keep source sample rates, shared start points, and channel layout across stems from the same recording. The recognizer consumes the prepared WAVs offline.

## Input catalog and permission record

Record each short excerpt as a JSON object on one line in `sources.jsonl`. Paths are relative to that file.

```json
{"song_id":"original-composition-101","recording_id":"session-4-take-2","clip_id":"verse-line-01","artist":"artist-014","genre":"folk","language":"en","annotation_verified":true,"alignment_verified":true,"text":"the rain has passed","start":4.2,"end":7.85,"vocal":"session4/vocal.wav","mix":"session4/original.wav","accompaniment":["session4/guitar.wav","session4/bass.wav"],"rights_id":"permission-14"}
```

- Assign the same `song_id` to every recording of a composition, including covers, edits, and remasters. Use one `artist` ID per lead singer across aliases and projects. Set `genre` to `unknown` when the source lacks a reliable genre label.
- Give each recording version a stable `recording_id`. Within a recording, use consistent source paths and nonoverlapping time intervals. Keep `clip_id` unique within its composition across versions.
- Listen for one lead singer delivering English lyrics. Check the transcript against the recording, confirm the excerpt boundaries by listening, and mark `annotation_verified` only after review. Supply start and end in seconds on the recording timeline, even when the clip is already cut.
- Supply each available `vocal` isolation and released `mix`, plus the complete set of non-vocal stems if available. Verify common timing by listening and checking export starts before marking `alignment_verified`. Keep absent conditions out of the catalog.

Record the underlying grant in a separate `rights.jsonl`. The curator verifies scope and keeps the source evidence.

```json
{"rights_id":"permission-14","source":"Local consent form for artist-014, dated YYYY-MM-DD","recording_permission":"Recording owner grants model training on the listed takes","lyrics_permission":"Lyricist owns the original lyric and grants model training","training_allowed":true,"checkpoint_release_allowed":false}
```

Use affirmative training permission for every rendered source. Record checkpoint release scope explicitly, including conditions in the underlying grant. A verified freeze binds the supplied evidence by digest. Retain the catalog and ledger locally so another reviewer can verify the actual permissions.

## Render and verify

Run from the repository root in the Python environment with NumPy installed.

```sh
python -m scripts.prepare_lyrics freeze data/cleared/sources.jsonl data/cleared/rights.jsonl runs/cleared-frozen --seed 0
python -m scripts.prepare_lyrics verify runs/cleared-frozen
```

The default maximum excerpt length is 30 seconds. Set `--max-seconds` when the study uses a different short-clip limit.

The automatic split groups every artist connected by a cover or shared composition. It assigns whole groups toward 80%, 10%, and 10% of excerpt duration, with deterministic tie-breaking. With fewer than three groups it fills train and then validation when possible. A curator can assign `split` on every source row to balance available genres and conditions. The program rejects explicit assignments that divide a connected group.

Preparation verifies equal WAV rates and frame lengths, checks that rounded excerpt bounds fit, checks for cross-split identical rendered WAVs, and checks zero-lag agreement between a supplied original and the sum of complete stems. The agreement threshold is 0.95 normalized correlation. A heavily mastered mix can fall outside that threshold, so document such a source and use only the conditions whose pairing can be verified. Partial stems rely on the curator's alignment review.

A full set emits seven matched conditions from the same interval.

| Condition | Waveform |
| --- | --- |
| `vocal` | Isolated vocal |
| `original_mix` | Supplied released mix |
| `mix_-12dB`, `mix_-6dB`, `mix_+0dB`, `mix_+6dB` | Isolated vocal with scaled accompaniment. Each number is accompaniment RMS relative to vocal RMS in that excerpt. |
| `instrumental` | Sum of the non-vocal stems with an empty reference lyric |

All conditions for an excerpt share the headroom scaling needed for PCM16. The prepared manifest includes the canonical artist and song IDs, recording and excerpt IDs, genre, rights ID, exact rendered duration, and condition. Count each vocal excerpt once for clean-vocal hours. When only vocals or mixes are available, the manifest contains those conditions alone. Silent backing contributes no controlled-mixture or instrumental condition.

The output directory must be new. Record the `manifest_sha256` from `freeze.json` before fitting or selecting thresholds, and use that same freeze for final scoring. Verify the frozen WAV and manifest bytes before each stage. The lock also records the seed, catalog and rights ledger digests, and whether all training sources permit checkpoint release.

Keep catalog text, audio, and any checkpoint with unresolved release rights in ignored local directories.
