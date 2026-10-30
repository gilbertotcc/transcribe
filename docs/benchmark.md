# Benchmark

This page describes how `transcribe` is benchmarked: the gates and the host,
the test material, the procedure, what each report contains, and the
measured results. The README repeats the headline numbers in its
[benchmark summary](../README.md#benchmark-summary). The meaning of each
accuracy metric is explained in
[How it works, section 3](how-it-works.md#3-what-the-benchmark-numbers-mean).

## 1. Gates and host

### Gates

The benchmark checks the pipeline against four gates. Every report marks
each gate as `passed`, `failed` or `unknown`. Gate targets change only with
the user's agreement, and each change is dated here.

| Gate                | Target                                         | Changes                                                                                                                              |
| ------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Runtime             | 60 min of audio in at most 60 min (RTF <= 1.0) |                                                                                                                                      |
| Memory              | Peak RSS and peak footprint <= 6 GB            | Relaxed from 4 GB to 4.5 GB, 2026-10-24; to 6 GB for `large-v3-turbo`, 2026-10-25; footprint added, so GPU memory counts, 2026-10-30 |
| WER                 | <= 8% on a hand-checked reference              |                                                                                                                                      |
| Speaker attribution | >= 90% of words correctly attributed           |                                                                                                                                      |

Speaker attribution is checked against the one-to-one score
([How it works, section 3.2](how-it-works.md#32-the-word-based-speaker-score)).

### Host

Two hosts. Results name the host they come from; until 2026-10-29 every
run used the first one, on the CPU only (measured 2026-10-24):

- 2020 13" MacBook Pro, Intel Core i5-1038NG7 @ 2.00 GHz, 4 cores / 8
  threads, x86_64.
- RAM: 16 GB. macOS 26.7.1 (25G241).
- Python 3.13.7, uv 0.13.5. Engine packages, all installed as binary wheels
  at the versions locked in `uv.lock`: `ctranslate2` 4.8.3, `av` 18.1.0,
  `onnxruntime` 1.23.2, `sherpa-onnx` 2.0.0, `tokenizers` 0.23.3,
  `faster-whisper` 1.2.1.

The second host runs the `apple-silicon` profile
([Apple Silicon](#apple-silicon)), measured 2026-10-29:

- 2022 13" MacBook Air, Apple M2, 8 cores (4 performance, 4 efficiency),
  8-core GPU, arm64.
- RAM: 16 GB, shared by CPU and GPU. macOS 26.5.2.
- Python 3.13.9, uv 0.13.6. The same engine packages as above, plus `mlx`
  0.32.4 and `mlx-whisper` 0.4.3.

## 2. Test material

### 2.1 Audio

The audio is the LibriVox recording of Oscar Wilde's *The Importance of Being
Earnest* (Version 2), a public-domain dramatic reading in which 7 readers
voice the characters ([LibriVox]). Each reader recorded their part
separately, so every change of speaker is also a change of microphone and
room (see [scope](how-it-works.md#34-scope-of-the-benchmark)).

The source files (one per act) and every clip cut from them stay in the
git-ignored `bench/` directory:

| Recording                        | Length | Content                                 | Role                            |
| -------------------------------- | ------ | --------------------------------------- | ------------------------------- |
| `clips/dialogue30.wav`           | 30 s   | Dialogue between two male voices        | Engine tests, quick checks      |
| `clips/silence10_dialogue30.wav` | 40 s   | The same dialogue after 10 s of silence | Leading-silence engine test     |
| `clips/act1_10min.wav`           | 10 min | Excerpt of Act 1                        | Representative recording        |
| `clips/acts1-2_60min.m4a`        | 60 min | The first hour of Acts 1 and 2          | Evaluation (gates)              |
| `source/earnest_03.mp3`          | 25 min | Act 3                                   | Held-out tuning                 |
| `clips/act2_tail_22min.wav`      | 22 min | The rest of Act 2, after the first hour | Held-out tuning (short replies) |

Settings are tuned on held-out audio (Act 3 and the tail of Act 2) and then
confirmed on the 60 min evaluation recording, so the gates are not measured
on the material used to choose the settings. The engine tests also accept
MP3 and M4A variants of the 30 s clip when they are present.

Two more sets of material check speaker diarization on Italian, without
speech recognition (see [section 2.3](#23-diarization-only-material)), and
one set checks speech recognition in Italian and English (see
[section 2.4](#24-multilingual-asr-material)):

| Recording                         | Length    | Content                                         | Role                         |
| --------------------------------- | --------- | ----------------------------------------------- | ---------------------------- |
| `clips/sim/sim_it_s{1,2,3}.wav`   | 20 min    | Simulated conversations, 6 Italian MLS readers  | Diarization, Italian         |
| `clips/sim/sim_en_s{1,2,3}.wav`   | 20 min    | The same construction, 6 English MLS readers    | Diarization, English control |
| `clips/asr/asr_it_s{1,2,3}.wav`   | 10 min    | Simulated, 6 Italian MLS readers, no cropping   | ASR and WER, Italian         |
| `clips/asr/asr_en_s{1,2,3}.wav`   | 10 min    | The same construction, 6 English MLS readers    | ASR and WER, English control |
| `source/locandiera/*_{1,2,3}.mp3` | 47-61 min | *La locandiera*, Acts 1-3, Italian cast reading | Diarization, cluster count   |

### 2.1.1 Rebuilding the material

`scripts/fetch_bench_material.py` recreates all of `bench/` on any host
from the committed recipe `scripts/bench_material.json`, which holds URLs,
sha256 hashes, cut points and the cast, but no audio and no text:

```sh
uv run scripts/fetch_bench_material.py
```

It downloads the three LibriVox acts, the play text (from a Project
Gutenberg mirror) and the two MLS splits, checks every hash, cuts the clips
in-process with PyAV, writes both references with `play_reference.py`, and
builds the 12 simulated conversations with `simulate_conversation.py`
(seeds 1-3). Existing files are kept, and a download is checked before it
is saved. `--with-locandiera` also fetches the three acts of
*La locandiera* (2.7 h), which only the cluster counts of section 2.3 use.
The 60 min recording is Act 1 followed by the first 22 min of Act 2;
`dialogue30.wav` is the exchange between Algernon and Lane at 2:13 of
Act 1.

### 2.2 Reference text

The references come from the Project Gutenberg edition of the play (#844)
([Wilde]), converted by `scripts/play_reference.py` into one line per
speech:

```text
READER<TAB>text of the speech
```

Each character is mapped to the reader who voiced it with repeated
`--cast CHARACTER=READER` options, because one reader may voice several
characters. Stage directions and scene descriptions get the reader `-`: they
count for WER but not for speaker attribution. There are two references:
`source/earnest_act1_2.tsv` (Acts 1 and 2) and `source/earnest_act3.tsv`
(Act 3).

The published text differs from what the readers said: archaic spellings
such as "any one", British spellings, and skipped stage directions all count
as recognition errors, so WER is slightly pessimistic. A spot check of the
10 min alignment attributes about 15 errors to the reference. A reference
may cover more than the recording; the scorer trims it to the span that
aligns with the recognized text.

### 2.3 Diarization-only material

**Simulated conversations.** `scripts/simulate_conversation.py` builds a
conversation from the test split of Multilingual LibriSpeech (MLS), read
LibriVox audiobooks with one reader per utterance ([Pratap et al., 2020]).
The splits come from Hugging Face, pinned to a revision, and stay in
`bench/source/mls/`:

| Language | Dataset and file                                              | Revision   | Speakers |
| -------- | ------------------------------------------------------------- | ---------- | -------- |
| Italian  | `facebook/multilingual_librispeech`, `italian/test-*.parquet` | `2e83e618` | 10       |
| English  | `parler-tts/mls_eng`, `data/test-*.parquet`                   | `faf6604d` | 42       |

With a fixed seed, the script picks 6 readers, takes each reader's utterances
from one chapter or book (one recording session), and lines up turns of
randomly chosen readers, never the same reader twice in a row. About a
quarter of the turns are cropped to 1-3 s. Gaps of 0.2-1.5 s separate the
turns, and no turns overlap. Every turn is scaled to the same RMS level. Both
languages use the same construction and the same seeds (1, 2, 3), so they
differ in language and readers only. The script also writes an RTTM file
with the exact turn times; it holds MLS speaker IDs and no text. A whole
utterance counts as speech in the RTTM, including its short pauses, so
part of the missed speech comes from the reference.

***La locandiera*.** The LibriVox recording of Carlo Goldoni's comedy is an
Italian cast reading: 9 readers voice 9 characters and a narrator, and
the recording has some sound effects ([LibriVox, Goldoni]). As in
*Earnest*, each reader recorded their part separately. The text is on
Wikisource ([Goldoni]). Counting the speeches there, every act has 6 or 7
speaking characters plus a servant and the narrator, so 8 or 9 readers.
There is no time-stamped reference, so only the cluster count is compared.

### 2.4 Multilingual ASR material

`simulate_conversation.py --reference FILE.tsv` builds the same kind of
conversation, 10 min long, and also writes each turn's MLS transcript as
`READER<TAB>text`, the format of section 2.2. A cropped turn would have no
known text, so with `--reference` no turn is cropped. Seeds 1, 2 and 3 give
three Italian and three English conversations. The reference files hold
text, so they stay in `bench/` with the audio.

MLS transcripts are lowercase, without punctuation, and spell numbers out,
while Whisper writes digits ([Pratap et al., 2020]). The scorer lowercases
and drops punctuation, but numbers still count as errors, so WER on this
material is an upper bound. It compares models on the same conversation; it
is not comparable with the *Earnest* WER. The scorer keeps Unicode letters
(`perché`, `l'uomo`) after NFC normalization.

## 3. Procedure

1. Provision the models once: `uv run scripts/provision_models.py`.
2. Run one recording per process, so that peak memory belongs to that run
   alone:

   ```sh
   uv run scripts/benchmark.py bench/clips/act1_10min.wav --label act1-10min \
       --reference bench/source/earnest_act1_2.tsv
   ```

3. To measure a different setting, override any `Settings` field with
   `--set FIELD=VALUE` (repeatable), for example
   `--set clustering_threshold=0.85`. The host's hardware profile is
   applied first; `--profile intel-mac` measures the Intel settings on any
   host, and `--profile apple-silicon` the GPU settings.
4. Compare settings one at a time, run back to back on the same host, and
   keep the other settings at their defaults.
5. Record agreed results in [section 5](#5-results), with the date of the
   run, and any resulting change of default in the
   [decision log](design.md#decision-log).

To score speaker turns without speech recognition, add `--diarization-only`
with an RTTM reference, or with the number of speakers when there is no
reference:

```sh
mkdir -p bench/clips/sim
uv run scripts/simulate_conversation.py bench/source/mls/mls_it_test.parquet \
    --speakers 6 --minutes 20 --seed 1 \
    --out bench/clips/sim/sim_it_s1.wav --rttm bench/clips/sim/sim_it_s1.rttm
uv run scripts/benchmark.py bench/clips/sim/sim_it_s1.wav --label sim-it-s1 \
    --diarization-only --rttm bench/clips/sim/sim_it_s1.rttm
```

To score recognition on a simulated conversation, write its reference with
`--reference` and pass that file to the benchmark:

```sh
mkdir -p bench/clips/asr
uv run scripts/simulate_conversation.py bench/source/mls/mls_it_test.parquet \
    --speakers 6 --minutes 10 --seed 1 --out bench/clips/asr/asr_it_s1.wav \
    --rttm bench/clips/asr/asr_it_s1.rttm --reference bench/clips/asr/asr_it_s1.tsv
uv run scripts/benchmark.py bench/clips/asr/asr_it_s1.wav --label asr-it-s1 \
    --reference bench/clips/asr/asr_it_s1.tsv
```

A speaker embedding model outside the manifest can be tried with an absolute
path, `--set embedding_model=/path/to/model.onnx`; the report then lists it
by file name and sha256.

Diarization experiments that should not change the shipped pipeline use
`scripts/diarization_lab.py`. Its `prepare` command decodes a recording and
runs ASR once. Later commands reuse that cache: `diarize` runs sherpa-onnx
with an optional `--num-clusters`, `embed` embeds turns, `sweep`
re-clusters and scores a grid of settings, and `diagnose-short` classifies
wrong short-reply words. Everything stays in `bench/lab/`, including the
experiment ledger `bench/lab/ledger.jsonl`.

Labels may contain only letters, digits, `.`, `_` and `-`; they are never
derived from the file name. `--alignment-out FILE` also writes the word
alignment, for checking the reference by hand. That file contains the
recognized text, so it stays out of the reports.

Runs so far were warm: the audio and models were already in the page cache.
Repeat runs match within timing noise. A truly cold run (`sudo purge` first)
has not been measured.

## 4. Reports

Each run writes `bench/reports/LABEL-TIMESTAMP.json`. A report holds
measurements only: no recognized text and no file paths.

| Field                             | Content                                                          |
| --------------------------------- | ---------------------------------------------------------------- |
| `host`                            | CPU, cores, RAM, OS, Python and engine package versions          |
| `models`, `settings`              | Model hashes and every `Settings` value used                     |
| `audio_seconds`, `stage_seconds`  | Audio length; seconds for decode, ASR, diarization and rendering |
| `real_time_factor`                | Total processing time divided by audio length                    |
| `peak_rss_bytes`                  | Peak resident memory of the process                              |
| `peak_gpu_bytes`                  | Peak MLX (GPU) allocation; 0 without MLX                         |
| `peak_footprint_bytes`            | Peak physical footprint (macOS), GPU memory included             |
| `profile`                         | Hardware profile used (`--profile`, default: this host's)        |
| `versus_intel_2020`               | RTF and speaker score against the Intel 60 min results           |
| `speakers_found`, `unknown_share` | Clusters found; share of words without a speaker                 |
| `wer`                             | WER with substitutions, deletions, insertions and word counts    |
| `speaker_accuracy`                | One-to-one and many-to-one scores, scored and unknown words      |
| `speaker_accuracy.by_turn_length` | The same scores for 1-2 word turns and for longer turns          |
| `gates`, `gate_targets`           | Status of each gate and the targets it was checked against       |
| `language`                        | Detected (or forced) language code and its probability           |
| `mode`                            | `full`, or `diarization-only` (no ASR, no WER, no word score)    |
| `diarization`                     | Diarization-only: DER parts, JER, purity, coverage, speakers     |

The memory gate reads `peak_footprint_bytes` when it exists. Resident set
size leaves out memory that MLX allocates for the GPU, and adding the two
peaks overstates, because they happen in different stages. The footprint
is the lifetime maximum that macOS charges the process
(`proc_pid_rusage`), which counts both. Intel reports up to 2026-10-26 have
only `peak_rss_bytes`; on the CPU path the two differ by a few percent.

`diarization` is computed with pyannote.metrics over the whole recording
([Bredin]). DER is reported at a collar of 0 and of 0.5 s; pyannote's collar
is the total width, so 0.5 s ignores 0.25 s on each side of every reference
boundary. Each DER part is in seconds: false alarm, missed speech and
confusion. In diarization-only mode the WER and speaker attribution gates
read `unknown`, because both are word-based.

`by_turn_length` treats each reference line as one turn. It scores both
groups with the cluster-to-reader maps of the whole recording, so a group's
score shows how its words fare under the same mapping as the overall score.

## 5. Results

All results below come from the host in [section 1](#host). Speaker
accuracy maps clusters to readers one-to-one (exact) and many-to-one; stage
directions are not scored.

### Baseline

Measured 2026-10-24 with the default `Settings` of that day, which used
`clustering_threshold` 0.5 (the value of the official sherpa-onnx example):

| Recording         | RTF  | ASR s | Diar. s | Peak RSS | Clusters | WER  | Speaker 1:1 | Many:1 |
| ----------------- | ---- | ----- | ------- | -------- | -------- | ---- | ----------- | ------ |
| 30 s dialogue WAV | 0.36 | 7.7   | 2.8     | 0.77 GB  | 3        | 6.7% | 86.5%       | 86.5%  |
| 10 min Act 1 WAV  | 0.27 | 117   | 43      | 1.17 GB  | 24       | 5.3% | 49.0%       | 88.5%  |
| 60 min Acts 1-2   | 0.30 | 643   | 424     | 4.34 GB  | 169      | 6.3% | 49.9%       | 93.7%  |

Gate status on the 60 min recording with the current defaults
(`clustering_threshold` 0.8, see [below](#clustering-threshold);
`large-v3-turbo`, see [turbo](#large-v3-turbo-and-selective-refinement);
re-clustering, see [folding](#folding-small-groups)), measured 2026-10-26:

| Gate                | 60 min result               |
| ------------------- | --------------------------- |
| Runtime             | passed (RTF 0.43)           |
| Memory              | passed (5.13 GB; gate 6 GB) |
| WER                 | passed (5.1%)               |
| Speaker attribution | passed (91.3%)              |

The default pipeline meets all four gates on the 60 min recording. Before
re-clustering (2026-10-25, RTF 0.48 and 4.58 GB), speaker attribution failed
at 74.1%: clusters were pure, but one reader was split across several
clusters, which the one-to-one score penalizes. With `small.en`
(2026-10-24, today's `--quick`) the run took RTF 0.30 and 3.36 GB, with
6.3% WER.

Because the runtime gate passed with a wide margin, the runtime levers
prepared in case it failed were not tried: `cpu_threads` variations,
`BatchedInferencePipeline` with batch size 2 and 4 (`asr_batch_size`), the
`base.en` model instead of `small.en`, and whisper.cpp as a second backend.
The margin was later spent on accuracy instead: `large-v3-turbo` became the
default recognizer ([turbo](#large-v3-turbo-and-selective-refinement)).

### Clustering threshold

Measured 2026-10-24. `clustering_threshold` swept on held-out Act 3 (25 min,
not used for the gates):

| Threshold | Clusters | Speaker 1:1 | Many:1 |
| --------- | -------- | ----------- | ------ |
| 0.5       | 97       | 48.9%       | 95.1%  |
| 0.6       | 68       | 55.7%       | 94.9%  |
| 0.7       | 47       | 60.6%       | 94.8%  |
| 0.8       | 27       | 82.0%       | 94.5%  |
| 0.9       | 18       | 82.8%       | 90.9%  |
| 1.0       | 9        | 84.6%       | 85.3%  |
| 1.1       | 4        | 81.5%       | 81.5%  |

At 0.9 and above, the two male voices of the 30 s clip merge into one speaker
(the engine test needs two). Agreed default: **0.8**. Results on the 60 min
evaluation recording:

| Threshold   | RTF  | Peak RSS | Clusters | WER  | Speaker 1:1 | Many:1 |
| ----------- | ---- | -------- | -------- | ---- | ----------- | ------ |
| 0.5         | 0.30 | 4.34 GB  | 169      | 6.3% | 49.9%       | 93.7%  |
| 0.8 default | 0.30 | 3.36 GB  | 40       | 6.3% | 74.1%       | 93.7%  |
| 1.0         | 0.33 | 3.29 GB  | 13       | 6.3% | 80.7%       | 86.2%  |

Raising `clustering_threshold` produces fewer clusters and a higher
one-to-one score, until different voices start to merge. Speaker attribution
stays **failed** against the 90% gate: clusters are pure (about 94%
many-to-one) but one reader still gets several labels. Peak memory varies
with the number of clusters.

### Conditioning on previous text

Measured 2026-10-24. `Settings.condition_on_previous_text`, default settings
otherwise. The 10 and 25 min rows ran in one batch; the 60 min `False` row
is the 0.8 default above.

| Recording       | Setting | ASR s | RTF  | WER  | Insertions |
| --------------- | ------- | ----- | ---- | ---- | ---------- |
| 10 min Act 1    | False   | 131   | 0.37 | 5.3% | 11         |
| 10 min Act 1    | True    | 268   | 0.61 | 4.8% | 12         |
| 25 min Act 3    | False   | 310   | 0.36 | 7.3% | 35         |
| 25 min Act 3    | True    | 730   | 0.66 | 6.4% | 29         |
| 60 min Acts 1-2 | False   | 650   | 0.30 | 6.3% | 117        |
| 60 min Acts 1-2 | True    | 1536  | 0.55 | 6.0% | 106        |

`True` lowers WER by 0.3 to 0.9 points but multiplies ASR time by 2.1 to
2.4; insertions show no repetition loops either way. Agreed default:
**False**, to keep headroom for older hardware.

### Short replies

Measured 2026-10-24. `speaker_accuracy.by_turn_length` splits the speaker
score into words of reference turns of 1-2 words ("Yes.", "Certainly not.")
and longer turns, under the whole-recording cluster maps. Tuning used Act 3
(25 short words) and a held-out clip, the last 1348 s of Act 2, which the
60 min recording does not cover (31 short words). Speaker accuracy,
one-to-one / many-to-one, short turns then longer turns:

| Setting                     | Act 3 short | Act 3 longer | Act 2 tail short | Act 2 tail longer |
| --------------------------- | ----------- | ------------ | ---------------- | ----------------- |
| defaults                    | 24% / 52%   | 82% / 95%    | 45% / 55%        | 75% / 91%         |
| `min_duration_on` 0.1       | 28% / 56%   | 82% / 95%    | 45% / 55%        | 75% / 91%         |
| `min_duration_on` 0.2       | 28% / 56%   | 82% / 95%    | 45% / 55%        | 75% / 91%         |
| `min_duration_off` 0.2      | 24% / 52%   | 82% / 95%    | 45% / 55%        | 75% / 91%         |
| `min_duration_off` 0.3      | 24% / 52%   | 82% / 95%    | 45% / 55%        | 75% / 91%         |
| `clustering_threshold` 0.75 | 20% / 48%   | 68% / 95%    | 45% / 55%        | 72% / 91%         |
| `clustering_threshold` 0.85 | 40% / 56%   | 86% / 95%    | 52% / 55%        | 76% / 91%         |

On the 60 min recording (defaults, 84 short words): 18% / 52% short against
75% / 94% longer.

Words in short replies are attributed far worse than words in longer turns.
Lowering `min_duration_on` to 0.1 or 0.2 lifts the Act 3 short-turn scores
by 4 points (one of 25 words) and leaves the Act 2 tail unchanged;
`min_duration_off` has no measurable effect, and a higher threshold helps
only partly. Even the best cluster-to-reader
map leaves about half of the short replies wrong, so over-clustering alone
does not explain this: the cause is more likely turn boundaries or word
timestamps that put a short reply inside a neighbouring turn. Defaults are
unchanged. `clustering_threshold` 0.85 helped overall but is untested on the
two-male-voice 30 s clip.

### Non-English diarization

Measured 2026-10-25, diarization only (no ASR), default `Settings` with
NeMo TitaNet small and `clustering_threshold` 0.8. The decision rule was
fixed before the runs: TitaNet counts as worse on Italian if, averaged over
the seeds, Italian DER (0.5 s collar) is more than 5 points above English,
or Italian has more than 50% more clusters per speaker.

Simulated conversations (6 speakers, 20 min each). The DER parts are shares
of the reference speech:

| Conversation | Clusters | DER (0 s) | DER (0.5 s) | Missed | False alarm | Confusion | JER   | Purity | Coverage |
| ------------ | -------- | --------- | ----------- | ------ | ----------- | --------- | ----- | ------ | -------- |
| Italian 1    | 15       | 23.6%     | 20.6%       | 5.8%   | 0.3%        | 14.6%     | 24.7% | 97.8%  | 76.8%    |
| Italian 2    | 12       | 28.9%     | 26.1%       | 5.7%   | 0.2%        | 20.3%     | 29.4% | 97.6%  | 71.5%    |
| Italian 3    | 16       | 28.0%     | 25.2%       | 10.1%  | 0.2%        | 14.9%     | 26.5% | 98.7%  | 72.4%    |
| English 1    | 17       | 27.5%     | 24.9%       | 5.5%   | 0.1%        | 19.4%     | 27.6% | 98.8%  | 72.8%    |
| English 2    | 31       | 54.4%     | 52.6%       | 10.6%  | 0.4%        | 41.5%     | 52.5% | 98.9%  | 46.1%    |
| English 3    | 24       | 44.8%     | 42.5%       | 9.0%   | 0.4%        | 33.0%     | 38.4% | 98.8%  | 55.8%    |
| Italian mean | 14.3     | 26.8%     | 24.0%       | 7.2%   | 0.2%        | 16.6%     | 26.9% | 98.1%  | 73.6%    |
| English mean | 24.0     | 42.3%     | 40.0%       | 8.4%   | 0.3%        | 31.3%     | 39.5% | 98.8%  | 58.2%    |

`clustering_threshold` on conversation 1 of each language:

| Threshold | Italian clusters | Italian DER (0.5 s) | English clusters | English DER (0.5 s) |
| --------- | ---------------- | ------------------- | ---------------- | ------------------- |
| 0.7       | 18               | 23.6%               | 23               | 36.7%               |
| 0.8       | 15               | 20.6%               | 17               | 24.9%               |
| 0.9       | 10               | 10.9%               | 11               | 19.1%               |

*La locandiera* against *Earnest*, same batch, cluster count only:

| Recording              | Length | Readers | Clusters | Clusters per reader |
| ---------------------- | ------ | ------- | -------- | ------------------- |
| *La locandiera*, Act 1 | 61 min | 8-9     | 48       | 5.3-6.0             |
| *La locandiera*, Act 2 | 47 min | 8-9     | 38       | 4.2-4.8             |
| *La locandiera*, Act 3 | 52 min | 8-9     | 37       | 4.1-4.6             |
| *Earnest*, 60 min      | 60 min | 7       | 40       | 5.7                 |
| *Earnest*, Act 3       | 25 min | 7       | 27       | 3.9                 |

Diarization-only runs took RTF 0.12-0.15; peak RSS was at most 1.97 GB
(*La locandiera*, Act 1).

The rule is not met: on Italian, TitaNet small did not do worse than on
English. Italian DER is 16 points *lower*, with fewer clusters per speaker;
the English spread across seeds (25-53%) is larger than the gap. Both
languages show the same pattern as the play: clusters are pure (purity
about 98%), but each speaker is split into 2-5 clusters (coverage 58-74%),
and confusion makes up most of the DER. The cast readings agree: *La
locandiera* gets about as many clusters per reader as *Earnest* for
recordings of similar length. Decision: keep TitaNet small, and do not
measure replacement models for language reasons; over-clustering is
language-independent and tracked in issue #14. On these simulated
conversations a threshold of 0.9 lowers DER in both languages, but it
merges the two male voices of the 30 s clip
([clustering threshold](#clustering-threshold)).

### Language detection and multilingual ASR

Measured 2026-10-25. Detection runs on the speech that VAD finds in the
first 10 min; it is reported with its probability, and a value at or below
0.5 triggers a warning ([How it works](how-it-works.md#22-speech-recognition-asrpy-settingspy)).

| Recording                              | Spoken  | Detected | Probability | Time  |
| -------------------------------------- | ------- | -------- | ----------- | ----- |
| 30 s dialogue after 10 s of silence    | English | English  | 0.88        | 2.9 s |
| 60 s Italian after 10 s of silence     | Italian | Italian  | 0.93        | 3.1 s |
| 60 s Italian after 10 s of white noise | Italian | Italian  | 0.93        | 2.9 s |
| First 3 s of the same Italian clip     | Italian | English  | 0.43        | 2.7 s |
| *La locandiera*, Act 1                 | Italian | Italian  | 0.99        | 4.3 s |
| *Earnest*, Act 3                       | English | English  | 0.99        | 4.2 s |

Every benchmark run in this section and the next also detected its language
correctly, with probabilities of 0.95-1.00. The only miss, 3 s of speech,
falls below the threshold, so the run warns and suggests `--language`.

Recognition with the `small` models (what `--quick` uses), in one batch.
"Multilingual" sends English to `small` instead of `small.en`:

| Recording              | English model | RTF       | ASR s   | Peak RSS     | WER   |
| ---------------------- | ------------- | --------- | ------- | ------------ | ----- |
| *Earnest*, 10 min      | `small.en`    | 0.36      | 121     | 1.26 GB      | 5.3%  |
| *Earnest*, 10 min      | multilingual  | 0.38      | 133     | 1.26 GB      | 6.3%  |
| *Earnest*, 60 min      | `small.en`    | 0.35      | 704     | 3.56 GB      | 6.4%  |
| English MLS, mean of 3 | `small.en`    | 0.32-0.40 | 106-151 | 1.18-1.60 GB | 13.0% |
| English MLS, mean of 3 | multilingual  | 0.31-0.39 | 100-120 | 1.29-1.30 GB | 12.2% |
| Italian MLS, mean of 3 | (`small`)     | 0.40-0.50 | 137-182 | 1.20-1.30 GB | 25.1% |

On the hand-checked *Earnest* reference, `small.en` is a point better than
the multilingual `small`; on the MLS conversations the two are within a
point, in the other direction. `--quick` keeps `small.en` for English.
Italian WER is about twice the English MLS WER with the same construction.
Detection adds 4-9 s per run. The 60 min run used 0.35 RTF and 3.56 GB,
against 0.30 and 3.36 GB in the first measurements; this batch also ran
diarization 25% slower than before, with unchanged code, so part of the
difference is machine load.

### Large-v3-turbo and selective refinement

Measured 2026-10-25. Two ways to use `large-v3-turbo` were compared with the
`small` models:

- **Turbo on everything**: turbo transcribes the whole recording.
- **Selective refinement**: `small` transcribes everything, then turbo
  decodes again the segments with low confidence (mean token log
  probability below -0.30, mean word probability below 0.80, or
  compression ratio above 2.4), and its words replace those of `small`
  there. Two variants were tried: the segments alone with 0.5 s of padding,
  capped at 30% of the recording ("narrow"), and each segment widened with
  its neighbours to a full 30 s window, capped at 30% or 50%.

WER on 10 min recordings:

| Recording      | `small` / `small.en` | Narrow refinement | Windows, 30% cap | Windows, 50% cap | Turbo on everything |
| -------------- | -------------------- | ----------------- | ---------------- | ---------------- | ------------------- |
| *Earnest*      | 5.3%                 | 5.8%              | 5.4%             | 5.2%             | **4.8%**            |
| Italian MLS 1  | 27.6%                | 23.4%             | 24.6%            | 24.2%            | **21.5%**           |
| Italian MLS 2  | 23.8%                | 20.2%             | 22.2%            | 19.6%            | **18.9%**           |
| English MLS 1  | 16.5%                | 15.3%             | 15.8%            | 14.9%            | **7.7%**            |
| Share refined  |                      | 8-34%             | 22-26%           | 35-41%           | 100%                |
| Extra ASR time |                      | 54-109 s          | 49-75 s          | 91-108 s         | 116-153 s           |

Extra ASR time is the turbo work per 10 min: the refinement stage, or turbo
minus `small` on the same recording. Peak RSS was 2.2-2.6 GB for every
turbo variant on 10 min, against 1.3 GB for `small` alone.

Turbo on everything had the lowest WER on every recording. Refinement
helped Italian, but on *Earnest* the narrow variant made WER worse: its
short clips cut words at their edges and gave turbo little context. Windows
fixed that, but at the cap where refinement starts to help (50%), turbo on
everything costs only 25-45 s more per 10 min and is more accurate.
Refinement also cannot repair speech that `small` skipped: on English MLS 1,
`small.en` deleted 89 reference words and turbo 23, and a word that is
never recognized has no segment to
flag. Whisper encodes 30 s windows, so a 4 s clip costs as much as a full
window, which keeps refinement from being much cheaper. Decision:
transcribe everything with turbo (D20); `--quick` keeps the `small` models
for speed (D21).

The 60 min evaluation recording with turbo on everything:

| Run                     | RTF  | Detection s | ASR s | Diar. s | Peak RSS | WER  | Speaker 1:1 |
| ----------------------- | ---- | ----------- | ----- | ------- | -------- | ---- | ----------- |
| `small.en` (`--quick`)  | 0.35 | 5.3         | 704   | 530     | 3.56 GB  | 6.4% | 74.1%       |
| Turbo on everything     | 0.42 | 12.5        | 1100  | 385     | 4.77 GB  | 5.1% | 74.1%       |
| Turbo, shipped defaults | 0.48 | 4.5         | 1271  | 461     | 4.58 GB  | 5.1% | 74.1%       |

The first turbo run also detected the language with turbo; the shipped
defaults detect with `small`. The two turbo runs recognize identically, so
the 15% gap in ASR time is machine load. Turbo adds 4-8 min per hour of
audio and 1.0-1.2 GB of peak memory; the memory gate was relaxed to 6 GB
for it (D23).

### Re-clustering and embedding models

Measured 2026-10-25 with `scripts/diarization_lab.py` (developer-only; the
shipped pipeline is unchanged). The lab keeps the sherpa-onnx turns of the
default settings and changes only their labels:

1. Each turn gets one speaker embedding, from 0.1 s inside its edges.
2. Turns of at least 2.5 s are anchors. They are grouped by weighted
   average linkage over cosine distance, which ignores sherpa's labels.
3. Every other turn joins the nearest group when its cosine similarity is
   at least 0.6; otherwise it keeps a label of its own.

sherpa-onnx clusters with complete linkage, which splits a voice as soon as
one pair of its embeddings is far apart. Average linkage needs most pairs to
be far apart.

Settings were chosen on development material only: Act 3, the Act 2 tail
and the six simulated conversations. The 30 s two-male-voice clip guards
against merging. About 400 configurations were screened, each in seconds
on cached turns and embeddings. The 60 min evaluation recording was run
only for the baseline and the two finalists. It was inspected before, so
it is not a pristine test set.

Columns are clusters and one-to-one speaker accuracy, or, on the simulated
conversations, clusters, DER (0.5 s collar) and purity. "Oracle count"
gives sherpa the true number of speakers (diagnostic only). A is ERes2Net
(`3dspeaker_speech_eres2net_sv_en_voxceleb_16k`, θ 0.40); B is NeMo
TitaNet large (θ 0.35); both use 2.5 s anchors and a 0.6 similarity.

| Recording       | Baseline (0.8)   | Oracle count    | A: ERes2Net      | B: TitaNet large |
| --------------- | ---------------- | --------------- | ---------------- | ---------------- |
| 30 s dialogue   | 2, 86.5%         |                 | 2, 86.5%         | 2, 86.5%         |
| Act 3           | 27, 81.9%        | 6, 81.6%        | 35, 92.0%        | 42, 89.8%        |
| Act 2 tail      | 20, 74.6%        | 5, 69.8%        | 23, 87.2%        | 27, 86.1%        |
| Italian 1       | 15, 20.6%, 97.8% | 6, 18.5%, 86.0% | 10, 10.5%, 97.8% | 10, 10.5%, 97.8% |
| Italian 2       | 12, 26.1%, 97.6% | 5, 17.8%, 86.7% | 6, 7.1%, 98.1%   | 8, 7.3%, 98.1%   |
| Italian 3       | 16, 25.2%, 98.7% | 6, 10.8%, 98.6% | 13, 13.5%, 98.8% | 16, 13.9%, 98.8% |
| English 1       | 17, 24.9%, 98.8% | 6, 23.4%, 84.5% | 16, 18.4%, 99.1% | 18, 19.4%, 99.1% |
| English 2       | 31, 52.5%, 98.9% | 6, 35.2%, 77.1% | 39, 36.4%, 98.9% | 41, 35.5%, 98.9% |
| English 3       | 24, 42.5%, 98.8% | 6, 23.8%, 86.7% | 20, 36.7%, 99.0% | 21, 40.4%, 99.1% |
| **60 min eval** | **40, 74.1%**    |                 | **48, 89.2%**    | **68, 87.5%**    |

Across the development material, A lifts the mean one-to-one score from
78.3% to 89.6%. It also lowers the mean DER of the simulated conversations
from 32.0% to 20.5%, with no loss of purity. Re-clustering with TitaNet
small, the shipped model, scored at most 79% one-to-one, so both the
linkage and the embedding model matter. Merging sherpa's clusters instead
of turns merged two readers of the Act 2 tail (many-to-one fell from 91%
to 70%).

The oracle count does not help. Complete linkage forced down to the true
number of speakers merges different voices: purity falls to 77-87%.

On the 60 min recording, A reaches 89.2%, against 74.1% for the baseline.
The speaker attribution gate (90%) is still **failed**. Turn embeddings
took 76 s with either model and re-clustering took 0.05 s, an estimated
+0.02 RTF. The embedding process alone peaked at 0.99 GB. Neither
candidate has run inside the shipped pipeline, so its full-run RTF and peak
RSS are not measured.

A side effect: many short, dissimilar turns keep labels of their own. A
has 48 clusters for 7 readers, most of them small. The one-to-one score
counts their words as wrong, but the transcript shows more labels.

Short replies (`diagnose-short`) classify each wrong short-reply word under
the one-to-one map. On the 60 min recording, A answers 31 of 84 short-reply
words correctly, against 16 for the baseline. Of the 53 still wrong:

- 35 have no turn of the right speaker within 0.5 s, so the reply was never
  segmented apart from the neighbouring speech;
- 16 sit in a small cluster outside the map;
- 2 are near a turn of the right speaker.

Boundary and timestamp errors are therefore rare, and attribution changes
cannot fix short replies. The cause is segmentation, which does not
separate a 1-2 word reply from the speaker around it.

### Folding small groups

Measured 2026-10-26, continuing the previous section. Re-clustering left
dozens of small groups, and the one-to-one score counts their words as
wrong. Folding gives each group with less than 40 s of speech the label of
its most similar larger group, when their cosine similarity is at least
0.5. Folding below 30-90 s, at similarities of 0.4-0.5, gave 92.1-92.7% on
the development material, so the result does not hinge on one setting.
Two other leads did not help:

- Starting from sherpa at threshold 0.7 instead of 0.8 scored at most
  92.5%.
- ERes2Net as sherpa's own embedding scored 66-74% on Act 3, against 81.9%
  for the baseline, at about twice the diarization time (calibrated at 0.7,
  0.8 and 0.9).

The finalists were chosen before the evaluation run: folding below 40 s
and below 60 s, both at 0.5. On the 60 min recording they gave identical
labels. The table shows the 40 s finalist, F, with the same columns as the
previous table:

| Recording       | Baseline (0.8)   | A: no folding    | F: folding       |
| --------------- | ---------------- | ---------------- | ---------------- |
| 30 s dialogue   | 2, 86.5%         | 2, 86.5%         | 2, 86.5%         |
| Act 3           | 27, 81.9%        | 35, 92.0%        | 15, 95.0%        |
| Act 2 tail      | 20, 74.6%        | 23, 87.2%        | 11, 90.4%        |
| Italian 1       | 15, 20.6%, 97.8% | 10, 10.5%, 97.8% | 9, 10.3%, 97.8%  |
| Italian 2       | 12, 26.1%, 97.6% | 6, 7.1%, 98.1%   | 6, 7.2%, 98.1%   |
| Italian 3       | 16, 25.2%, 98.7% | 13, 13.5%, 98.8% | 9, 12.2%, 98.8%  |
| English 1       | 17, 24.9%, 98.8% | 16, 18.4%, 99.1% | 10, 17.1%, 99.1% |
| English 2       | 31, 52.5%, 98.9% | 39, 36.4%, 98.9% | 31, 33.0%, 98.9% |
| English 3       | 24, 42.5%, 98.8% | 20, 36.7%, 99.0% | 12, 34.4%, 99.0% |
| **60 min eval** | **40, 74.1%**    | **48, 89.2%**    | **18, 91.3%**    |

On the 60 min recording, F reaches **91.3%**, above the 90% target of the
speaker attribution gate. Many-to-one accuracy is 93.9%, the same as the
lab baseline. Each reader keeps a label of their own, with 92-99% of its
words from that reader. One smaller label (465 words) mixes two readers
almost evenly (54% and 45%); it forms without folding too, and the baseline
had a similar one (74% and 24%). In the short-reply group
F gets 42% right (baseline 19%); 37 of the 49 words still wrong have no
turn of the right speaker nearby, as in the previous section.

On *La locandiera*, never used for tuning, F leaves 18, 20 and 13 clusters
for the three acts (baseline 48, 38, 37), with 8-9 readers per act. There
is no time-stamped reference, so a merge there would go unnoticed.

**In the pipeline.** F became the default (D27). It ran through
`scripts/benchmark.py` on the 60 min recording on 2026-10-26:

| Run                               | RTF  | ASR s | Diar. s | Peak RSS | Clusters | WER  | Speaker 1:1 | Many:1 |
| --------------------------------- | ---- | ----- | ------- | -------- | -------- | ---- | ----------- | ------ |
| Before re-clustering (2026-10-25) | 0.48 | 1271  | 461     | 4.58 GB  | 40       | 5.1% | 74.1%       | 93.7%  |
| With re-clustering (2026-10-26)   | 0.43 | 1085  | 443     | 5.13 GB  | 18       | 5.1% | 91.3%       | 93.9%  |

The pipeline reproduces the lab score exactly. The diarization stage now
includes the turn embeddings. Its time stayed within the run-to-run
variation of this host, which also explains the lower ASR time with
unchanged ASR code. Peak RSS rose by 0.55 GB. Short replies score 42%
one-to-one (84 words), and longer turns 92%.

Folding has a risk that this material cannot test: a speaker who talks
for less than 40 s in total and sounds like someone else can be folded
into that speaker.

### Apple Silicon

Measured 2026-10-29 and 2026-10-30 on the second [host](#host), on the
rebuilt material ([2.1.1](#211-rebuilding-the-material)). The question was
which recognizer a newer Mac should use, under the condition that it does at
least as well as the 2020 Intel MacBook Pro. Four candidates were run on the
held-out recordings first:

- **C0**: the Intel settings (`--profile intel-mac`): faster-whisper
  `large-v3-turbo`, int8, beam 5, on the CPU, 4 threads.
- **C1**: C0 with 8 threads.
- **G1**: the same checkpoint through mlx-whisper on the GPU, fp16, greedy.
- **G2**: full `large-v3` through mlx-whisper on the GPU, fp16, greedy.

WER, then RTF (peak memory, RSS plus GPU, in brackets for the long
recordings). Speaker scores were the same for all four within 1 point on
every recording, and C1 matched C0 in WER exactly:

| Recording     | C0                  | C1   | G1                          | G2                  |
| ------------- | ------------------- | ---- | --------------------------- | ------------------- |
| Act 3         | 6.4%, 0.38 (2.4 GB) | 0.38 | 6.6%, **0.21** (3.9 GB)     | 6.7%, 0.35 (6.2 GB) |
| Act 2 tail    | 5.4%, 0.41 (2.7 GB) | 0.47 | **5.2%**, **0.21** (5.4 GB) | 5.6%, 0.34 (8.0 GB) |
| English MLS 1 | 7.7%, 0.47          | 0.44 | 7.6%, **0.22**              | **6.8%**, 0.32      |
| English MLS 2 | 6.7%, 0.43          | 0.41 | 6.0%, **0.20**              | **5.3%**, 0.30      |
| English MLS 3 | 6.6%, 0.46          | 0.51 | 5.8%, **0.23**              | **5.6%**, 0.32      |
| Italian MLS 1 | 20.9%, 0.45         | 0.43 | 16.6%, **0.22**             | **15.6%**, 0.32     |
| Italian MLS 2 | 15.8%, 0.44         | 0.42 | **15.3%**, **0.21**         | 18.2%, 0.33         |
| Italian MLS 3 | 19.0%, 0.51         | 0.44 | 17.6%, **0.21**             | **12.9%**, 0.32     |

The M2's CPU alone is barely faster than the Intel host (C0 RTF 0.38-0.51
against 0.43), and more threads did not help, because half of the eight
cores are efficiency cores. On the GPU, the same checkpoint (G1) ran twice
as fast. Its WER was lower on 7 of 8 recordings, by up to 4.3 points on
Italian, and 0.2 points higher on Act 3. Greedy decoding therefore cost
nothing on average against beam 5. Full `large-v3` (G2) was more accurate on
some recordings and less on others, ran slower than G1 and needed
6.2-8.1 GB, above the memory gate, so it was dropped.

Diarization runs on the CPU in every candidate. The CoreML provider of
sherpa-onnx, tried with G1 on Act 3, gave the same speaker score but took
301 s instead of 152 s and peaked at 8.2 GB RSS, so diarization stays on
the CPU (D31).

**Memory.** The first G1 run on the 60 min recording peaked at a physical
footprint of 11.1 GB, although MLX reported a peak of 3.3 GB in use: MLX
keeps freed GPU buffers in a cache, which grows toward the GPU working-set
limit. The adapter now caps that cache at 512 MB and releases the model
before diarization. On Act 3 this brought the footprint to 3.4 GB (C0:
4.2 GB) with the same RTF.

On the 60 min evaluation recording, C0 and G1 against the Intel results of
2026-10-26:

| Run                      | RTF  | ASR s | Diar. s | Peak memory | WER   | Speaker 1:1 | Many:1 |
| ------------------------ | ---- | ----- | ------- | ----------- | ----- | ----------- | ------ |
| Intel, original cut (C0) | 0.43 | 1085  | 443     | 5.13 GB RSS | 5.1%  | 91.3%       | 93.9%  |
| M2, C0, rebuilt cut      | 0.38 | 1035  | 331     | 5.54 GB RSS | 6.32% | 93.5%       | 93.6%  |
| M2, G1 (`apple-silicon`) | 0.20 | 366   | 341     | 4.07 GB     | 6.45% | 93.6%       | 93.7%  |

The G1 row is the final run with `--profile auto` (2026-10-30); the G1
peak is the physical footprint, GPU memory included. GPU decoding is not
bit-for-bit repeatable: four G1 runs on this recording gave 6.39-6.45% WER
and the same speaker score. On the same audio, G1 matches C0 in speaker
score, and its WER is 0.07-0.13 points higher (8-14 of 11,010 reference
words). The 1.2-point gap to the Intel WER is the
rebuilt material, not the host: C0 runs the Intel settings and shows the
same gap. G1 is therefore at least as good as the Intel reference in
runtime, speaker score and memory, and within 0.13 points in WER, while
being lower on most held-out recordings. It became the `apple-silicon`
profile (D29). Gate status of that profile on the 60 min recording:

| Gate                | 60 min result, `apple-silicon` |
| ------------------- | ------------------------------ |
| Runtime             | passed (RTF 0.20)              |
| Memory              | passed (4.07 GB; gate 6 GB)    |
| WER                 | passed (6.45%)                 |
| Speaker attribution | passed (93.6%)                 |

## 6. Limitations

- Two hosts, one per profile: an M1, an M3 or a Mac with 8 GB has not been
  measured. The hand-checked WER and the word-based speaker score come from
  one English play; Italian WER comes from simulated conversations of read
  speech, scored against MLS transcripts (section 2.4).
  Italian diarization was measured without ASR, on simulated conversations
  of read speech (no overlap, 6 speakers, no control of gender) and on the
  cluster count of one cast reading. Accuracy on real meetings, with shared
  microphones and overlapping speech, is still to be measured.
- The play has few short replies: the two held-out recordings together have
  56 short-turn words, so small differences in that group are not
  significant.
- The reference introduces some WER on its own (section 2.2).
- Timings come from warm runs and vary with machine load; compare settings
  only within one batch.

## References

[Bredin] Bredin, H. *pyannote.metrics: Reference*. Retrieved from
<https://pyannote.github.io/pyannote-metrics/reference.html>

[Goldoni] Goldoni, C. *La locandiera*. Wikisource.
<https://it.wikisource.org/wiki/La_locandiera>

[LibriVox] LibriVox. *The importance of being earnest (version 2) by Oscar
Wilde* [Audiobook]. Retrieved from
<https://librivox.org/the-importance-of-being-earnest-version-2-by-oscar-wilde/>

[LibriVox, Goldoni] LibriVox. *La locandiera by Carlo Goldoni*
[Audiobook]. Retrieved from
<https://librivox.org/la-locandiera-by-carlo-goldoni/>

[Pratap et al., 2020] Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., &
Collobert, R. (2020). MLS: A large-scale multilingual dataset for speech
research. *Proc. Interspeech 2020*, 2757-2761.
<https://doi.org/10.21437/Interspeech.2020-2826>

[Wilde] Wilde, O. *The importance of being earnest: A trivial
comedy for serious people* (Project Gutenberg eBook #844). Project Gutenberg.
<https://www.gutenberg.org/ebooks/844>

[Bredin]: https://pyannote.github.io/pyannote-metrics/reference.html
[Goldoni]: https://it.wikisource.org/wiki/La_locandiera
[LibriVox]: https://librivox.org/the-importance-of-being-earnest-version-2-by-oscar-wilde/
[LibriVox, Goldoni]: https://librivox.org/la-locandiera-by-carlo-goldoni/
[Pratap et al., 2020]: https://doi.org/10.21437/Interspeech.2020-2826
[Wilde]: https://www.gutenberg.org/ebooks/844
