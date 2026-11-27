# Training data and rights

## Vocadito English subset

The released ChorusTrace checkpoint was trained from random initialization on eight reviewed English Vocadito excerpts. ChorusTrace prepared mono PCM clips and extracted NumPy log-mel features before fitting. Vocadito is by Rachel M. Bittner, Katherine Pasalo, Juan José Bosch, Gabriel Meseguer Brocal, and David Rubinstein and is licensed under CC BY 4.0. [Checkpoint](../benchmarks/checkpoints/vocadito_long_context.npz) · [Vocadito release](https://zenodo.org/records/5557945) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Vocadito collected short, monophonic singing excerpts from volunteers who agreed to anonymized public release. The authors asked volunteers for original or public-domain compositions, manually edited the clip edges, and commissioned human lyric and note annotations. The official deposit licenses these recordings and annotations under CC BY 4.0. The composition review below covers the underlying words separately. [Official release](https://zenodo.org/records/5557945) · [Technical report](https://arxiv.org/abs/2110.05580) · [License](https://creativecommons.org/licenses/by/4.0/)

The recording and annotation license permits use and adaptation for recognizer training. The approved tracks contain traditional or public-domain lyrics. The ledger allows distribution of a trained recognition checkpoint with attribution under the CC BY 4.0 grant and records that determination independently of the source audio. Vocadito's publication supplies the dataset license and scope. Creative Commons explains how its existing licenses apply to AI training. [Creative Commons guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training/)

Keep audio, full lyric manifests and unreleased checkpoints in the ignored local output directories.

| Track ID | Canonical composition | Evidence for the lyrics |
| --- | --- | --- |
| 10, 34 | Rattlin’ Bog | Traditional variants, both in the Roud no. 129 song family. [Gloucestershire Traditions folk-song archive](https://glostrad.com/green-grass-grew-all-around-the/) |
| 11 | Oh, My Darling Clementine | Percy Montrose's 1884 publication. [Morgan Library](https://www.themorgan.org/music-manuscripts-and-printed-music/140365) |
| 14 | Happy Birthday to You | Early publication and public-domain settlement. [WIPO](https://www.wipo.int/en/web/wipo-magazine/articles/in-the-courts-court-confirms-legal-status-of-happy-birthday-to-you-55581) |
| 15 | House of the Rising Sun | Traditional US folk lyrics. The Folkways notes document early versions that predate the well-known commercial arrangements. [Smithsonian Folkways liner notes](https://folkways-media.si.edu/docs/folkways/artwork/FW02363.pdf) |
| 19 | Take Me Out to the Ball Game | Jack Norworth and Albert von Tilzer's 1908 sheet music. [Library of Congress](https://www.loc.gov/item/2005561767/) |
| 20 | We Wish You a Merry Christmas | Traditional English refrain, documented in the 1935 publication. This is a solo performance of the refrain. [Historical score](https://imslp.org/wiki/We_Wish_You_a_Merry_Christmas_(Warrell,_Arthur_Sydney)) |
| 21 | Drunken Sailor | Traditional sea shanty. [Library of Congress](https://www.loc.gov/item/afc9999005.12828/) |
| 32 | Baa, Baa, Black Sheep | Nursery rhyme documented in *Tommy Thumb’s Pretty Song Book* (1744). [University of Toronto](https://rpo.library.utoronto.ca/content/bah-bah-black-sheep) |
| 38 | Up on the Housetop | Benjamin Hanby's 1864 composition. [Historical score](https://imslp.org/wiki/Up_on_the_Housetop_(Hanby,_Benjamin_Russel)) |
| 39 | Hello! Ma Baby | Joseph E. Howard and Ida Emerson's 1899 sheet music. [Library of Congress](https://www.loc.gov/item/99470294/) |

The release contains 17 English tracks. Tracks 12, 33, and 40 contain identifiable modern commercial lyrics with no separate permission in the release. Tracks 6 and 18 lack individual composition provenance, and track 22 uses the modern wording of a traditional rhyme whose earlier documented wording differs. These six tracks were excluded. The acquisition command checks the pinned archive SHA-256 and exports only the approved recordings and labels.

```sh
.venv/bin/python -m scripts.acquire_vocadito data/cleared/vocadito
.venv/bin/python -m scripts.prepare_lyrics freeze \
  data/cleared/vocadito/sources.jsonl data/cleared/vocadito/rights.jsonl \
  runs/vocadito-frozen --seed 0 --max-seconds 32
.venv/bin/python -m scripts.prepare_lyrics verify runs/vocadito-frozen
```

The approved excerpts total **225.173605 seconds (0.062548224 hours)**, from 10 compositions and 10 anonymized singer IDs. Human labels cover each supplied excerpt from start to end, including any short edge silence left by the authors. The importer derives end times from WAV frame counts. Genre labels are assigned from the documented composition families. Mean sung pitches per excerpt span MIDI 49–61. There are no original accompanied mixes or accompaniment stems in this release.

| Split | Excerpts | Song groups | Singer IDs | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Train | 8 | 7 | 7 | 179.711791 |
| Validation | 1 | 1 | 1 | 20.280000 |
| Test | 2 | 2 | 2 | 25.181814 |

## Children's Song Dataset research diagnostic

The CSD authors recorded one Korean professional pop singer in a soundproof room singing each song in two keys. They designed the corpus for singing-voice research, including training singing-voice models, and licensed the recordings and annotations under CC BY-NC-SA 4.0. The per-composition ledger permits local noncommercial training on the approved recordings and lyrics. Checkpoints from this arm are kept private because a separate permission covering checkpoint release was unavailable. [Official release](https://zenodo.org/records/4785016) · [License and dataset documentation](https://github.com/equal-singer/CSD#6-license) · [Technical report](https://program.ismir2020.net/static/lbd/ISMIR2020-LBD-435-abstract.pdf)

The importer selects 19 of the 50 English compositions, including only the documented traditional or public-domain stanzas where a recording has additional verses. The allowlist contains the composition sources and selected line ranges. For each released WAV it maps the human lyric lines through the annotated phonetic syllables to the exact CSV note sequence, and uses those note times for the excerpt boundaries. The original stereo PCM WAV appears once per recording in the source cache. Excerpts use disjoint frame intervals. [CSD preparation code](../scripts/acquire_csd.py)

The 38 released renditions yield **140 excerpts, 2158.807415 seconds (0.599668726 hours)**. The first-key renditions account for 1078.293855 seconds (0.299526071 hours). The 44 carol, 60 folk and 36 nursery-rhyme excerpts have one singer and no released accompaniment or original accompanied mixes. The strict corpus freezer assigns every row to training. CSD and vocadito report anonymous singers with different metadata, so the two releases cannot establish cross-source performer independence. CSD is kept outside vocadito's held-out artist benchmark. The overlapping CSD composition *We Wish You a Merry Christmas* matches a vocadito training song. CSD's *Clementine* is withheld because that song appears in the vocadito test split.

Download the official archive outside the recognizer and check the MD5 listed in the deposit, `4c586caff29ca05264cf7223cad1b8a2`. The local experiment used HTTP byte ranges from that release and verified the official zip CRC of each selected member. The same importer accepts the complete archive.

```sh
mkdir -p data/cleared/csd
curl -fLsS https://zenodo.org/records/4785016/files/CSD.zip -o data/cleared/csd/CSD.zip
md5 -q data/cleared/csd/CSD.zip
.venv/bin/python -m scripts.acquire_csd data/cleared/csd/CSD.zip data/cleared/csd/sources
.venv/bin/python -m scripts.prepare_lyrics freeze \
  data/cleared/csd/sources/sources.jsonl data/cleared/csd/sources/rights.jsonl \
  runs/csd-training-only --seed 0 --max-seconds 32
.venv/bin/python -m scripts.prepare_lyrics verify runs/csd-training-only
```

## Other sources reviewed

| Source | Evidence and remaining gap |
| --- | --- |
| MUSDB18 | The educational-use dataset lacks confirmed contributor clearance for this training. Prior artifacts remain outside these experiments. [Release terms](https://zenodo.org/records/1117372) |
| Cadenza CLIP | Human-labeled English excerpts with per-track Free Music Archive licenses. Its README defers to the original-file terms, and the challenge paper describes transcriptions drawn from both listening and published lyrics. Rights to those underlying words still require track-level verification. Available clean estimates use source separation on mixtures. [Release](https://zenodo.org/records/17950664) · [Data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12861282/) |
| URSing | The deposit licenses the recorded performances for research and includes synchronized stems. The track catalog lists contemporary commercial covers and references, without matching lyric or accompaniment grants. [Release and manual](https://zenodo.org/records/6404999) |
| NHSS | The end-user agreement requires a signed grant for internal noncommercial research. The described material includes English pop-song lyrics. No signed grant was obtained. [Agreement](https://hltnus.github.io/NHSSDatabase/uploads/NUSLicence.pdf) |
| GTSinger | The published CC BY-NC-SA 4.0 terms include a reservation around copyrighted audio. The English archive still needs track-level composition and recording checks. [Release](https://huggingface.co/datasets/GTSinger/GTSinger) · [Dataset license](https://huggingface.co/datasets/GTSinger/GTSinger/blob/main/dataset_license.md) |
| SingStyle111 | The paper identifies professionally recorded clean English singing, including songs with lyrics found online. The paper's license identifies its text. Further evidence is needed for recording and composition use. [Paper](https://www.cs.cmu.edu/~rbd/papers/singstyle2023.pdf) |
| VocalSet | Released professional recordings cover vowels and vocal techniques. The exercised sounds supply no usable English lyric labels for this benchmark. [Dataset](https://zenodo.org/records/1442513) |

Consenting contributors can expand the corpus by supplying solo performances of lyrics they own or a verified public-domain text. A useful per-recording grant identifies the recording owner, singer, lyricist and backing-track owner. It grants recognizer training and evaluation, and separately records whether model checkpoints, text references, and audio excerpts may be distributed. Capture a stable pseudonymous artist ID and canonical work ID at intake so repeat performances can be grouped and cross-dataset artist overlap can be checked.
