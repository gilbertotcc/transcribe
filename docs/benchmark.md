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

| Gate                | Target                                         | Changes                       |
| ------------------- | ---------------------------------------------- | ----------------------------- |
| Runtime             | 60 min of audio in at most 60 min (RTF <= 1.0) |                               |
| Memory              | Peak RSS <= 4.5 GB                             | Relaxed from 4 GB, 2026-10-24 |
| WER                 | <= 8% on a hand-checked reference              |                               |
| Speaker attribution | >= 90% of words correctly attributed           |                               |

Speaker attribution is checked against the one-to-one score
([How it works, section 3.2](how-it-works.md#32-the-word-based-speaker-score)).

### Host

All runs so far used one host, on the CPU only (measured 2026-10-24):

- 2020 13" MacBook Pro, Intel Core i5-1038NG7 @ 2.00 GHz, 4 cores / 8
  threads, x86_64.
- RAM: 16 GB. macOS 26.7.1 (25G241).
- Python 3.13.7, uv 0.13.5. Engine packages, all installed as binary wheels
  at the versions locked in `uv.lock`: `ctranslate2` 4.8.3, `av` 18.1.0,
  `onnxruntime` 1.23.2, `sherpa-onnx` 2.0.0, `tokenizers` 0.23.3,
  `faster-whisper` 1.2.1.

## 2. Test material

### 2.1 Audio

The audio is the LibriVox recording of Oscar Wilde's *The Importance of Being
Earnest* (Version 2), a public-domain dramatic reading in which 7 readers
voice the characters ([LibriVox]). Each reader recorded their part
separately, so every change of speaker is also a change of microphone and
room (see [scope](how-it-works.md#34-scope-of-the-benchmark)).

The source files (one per act) and every clip cut from them stay in the
git-ignored `bench/` directory:

| Recording                           | Length | Content                                      | Role                            |
| ----------------------------------- | ------ | -------------------------------------------- | ------------------------------- |
| `clips/dialogue30.wav`              | 30 s   | Dialogue between two male voices             | Engine tests, quick checks      |
| `clips/silence10_dialogue30.wav`    | 40 s   | The same dialogue after 10 s of silence      | Leading-silence engine test     |
| `clips/act1_10min.wav`              | 10 min | Excerpt of Act 1                             | Representative recording        |
| `clips/acts1-2_60min.m4a`           | 60 min | The first hour of Acts 1 and 2               | Evaluation (gates)              |
| `source/earnest_03.mp3`             | 25 min | Act 3                                        | Held-out tuning                 |
| `clips/act2_tail_22min.wav`         | 22 min | The rest of Act 2, after the first hour      | Held-out tuning (short replies) |

Settings are tuned on held-out audio (Act 3 and the tail of Act 2) and then
confirmed on the 60 min evaluation recording, so the gates are not measured
on the material used to choose the settings. The engine tests also accept
MP3 and M4A variants of the 30 s clip when they are present.

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
   `--set clustering_threshold=0.85`.
4. Compare settings one at a time, run back to back on the same host, and
   keep the other settings at their defaults.
5. Record agreed results in [section 5](#5-results), with the date of the
   run, and any resulting change of default in the
   [decision log](design.md#decision-log).

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

| Field                                    | Content                                                          |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `host`                                   | CPU, cores, RAM, OS, Python and engine package versions          |
| `models`, `settings`                     | Model hashes and every `Settings` value used                     |
| `audio_seconds`, `stage_seconds`         | Audio length; seconds for decode, ASR, diarization and rendering |
| `real_time_factor`                       | Total processing time divided by audio length                    |
| `peak_rss_bytes`                         | Peak resident memory of the process                              |
| `speakers_found`, `unknown_share`        | Clusters found; share of words without a speaker                 |
| `wer`                                    | WER with substitutions, deletions, insertions and word counts    |
| `speaker_accuracy`                       | One-to-one and many-to-one scores, scored and unknown words      |
| `speaker_accuracy.by_turn_length`        | The same scores for 1-2 word turns and for longer turns          |
| `gates`, `gate_targets`                  | Status of each gate and the targets it was checked against       |

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
(`clustering_threshold` 0.8, see [below](#clustering-threshold)):

| Gate                | 60 min result                                         |
| ------------------- | ----------------------------------------------------- |
| Runtime             | passed (RTF 0.30)                                     |
| Memory              | passed (3.36 GB at the 0.8 default)                   |
| WER                 | passed (6.3%)                                         |
| Speaker attribution | **failed** (74.1% at the 0.8 default)                 |

The default pipeline meets the runtime, memory and WER gates on the 60 min
recording, with a wide margin on runtime. Speaker attribution fails:
clusters are pure, but one reader is split across several clusters, which
the one-to-one score penalizes.

Because the runtime gate passed with a wide margin, the runtime levers
prepared in case it failed were not tried: `cpu_threads` variations,
`BatchedInferencePipeline` with batch size 2 and 4 (`asr_batch_size`), the
`base.en` model instead of `small.en`, and whisper.cpp as a second backend.

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
Changing `min_duration_on` and `min_duration_off` has no measurable effect,
and a higher threshold helps only partly. Even the best cluster-to-reader
map leaves about half of the short replies wrong, so over-clustering alone
does not explain this: the cause is more likely turn boundaries or word
timestamps that put a short reply inside a neighbouring turn. Defaults are
unchanged. `clustering_threshold` 0.85 helped overall but is untested on the
two-male-voice 30 s clip.

## 6. Limitations

- One play, one host, one language. Accuracy on real meetings, with shared
  microphones and overlapping speech, is still to be measured.
- The play has few short replies: the two held-out recordings together have
  56 short-turn words, so small differences in that group are not
  significant.
- The reference introduces some WER on its own (section 2.2).
- Timings come from warm runs and vary with machine load; compare settings
  only within one batch.

## References

[LibriVox] LibriVox. *The importance of being earnest (version 2) by Oscar
Wilde* [Audiobook]. Retrieved from
<https://librivox.org/the-importance-of-being-earnest-version-2-by-oscar-wilde/>

[Wilde] Wilde, O. *The importance of being earnest: A trivial
comedy for serious people* (Project Gutenberg eBook #844). Project Gutenberg.
<https://www.gutenberg.org/ebooks/844>

[LibriVox]: https://librivox.org/the-importance-of-being-earnest-version-2-by-oscar-wilde/
[Wilde]: https://www.gutenberg.org/ebooks/844
