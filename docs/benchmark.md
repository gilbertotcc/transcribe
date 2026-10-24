# Benchmark

This page describes how `transcribe` is benchmarked: the test material, the
procedure, what each report contains, and what the experiments so far have
shown. The measured numbers live in section 9 of the
[specification][spec-9] and, in summary, in the [README](../README.md#benchmark-summary).
The meaning of each accuracy metric is explained in
[How it works, section 3](how-it-works.md#3-what-the-benchmark-numbers-mean).

## 1. Goals and gates

The benchmark checks the MVP against four gates defined in section 9 of the
specification: runtime (real-time factor), peak memory, word error rate (WER),
and speaker attribution. Every report marks each gate as `passed`, `failed`
or `unknown`. Gate targets change only with the user's agreement, and each
change is dated in the specification.

All runs so far used one host: the 2020 Intel MacBook Pro described in
section 2 of the specification, on the CPU only.

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
as recognition errors, so WER is slightly pessimistic. A reference may cover
more than the recording; the scorer trims it to the span that aligns with
the recognized text.

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
5. Record agreed results and decisions in section 9 of the specification.

Labels may contain only letters, digits, `.`, `_` and `-`; they are never
derived from the file name. `--alignment-out FILE` also writes the word
alignment, for checking the reference by hand. That file contains the
recognized text, so it stays out of the reports.

Runs so far were warm: the audio and models were already in the page cache.
A truly cold run (`sudo purge` first) has not been measured.

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

## 5. Experiments and findings

Each experiment below links to its tables in section 9 of the specification.

**[Baseline][spec-baseline] (M5).** The default pipeline meets the runtime,
memory and WER gates on the 60 min recording, with a wide margin on runtime.
Speaker attribution fails: clusters are pure, but one reader is split across
several clusters, which the one-to-one score penalizes.

**[Clustering threshold][spec-threshold] (M5).** Raising
`clustering_threshold` produces fewer clusters and a higher one-to-one score,
until different voices start to merge: from 0.9 upwards, the two male voices
of the 30 s clip become one speaker. The default is 0.8. Peak memory varies
with the number of clusters.

**[Conditioning on previous text][spec-conditioning].** Enabling
`condition_on_previous_text` lowers WER slightly on every recording but more
than doubles ASR time; no repetition loops appeared either way. The default
stays `False` to keep headroom for older hardware.

**[Short replies][spec-short].** Words in 1-2 word turns ("Yes.", "Certainly
not.") are attributed far worse than words in longer turns. Changing
`min_duration_on` and `min_duration_off` has no measurable effect, and a
higher threshold helps only partly. Even the best cluster-to-reader map
leaves about half of the short replies wrong, so the cause is more likely
turn boundaries or word timestamps than over-clustering. Section 12 of the
specification lists the follow-up.

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
[spec-9]: ../idea/transcribe-mvp-agent-plan.md#9-benchmark
[spec-baseline]: ../idea/transcribe-mvp-agent-plan.md#baseline-results-m5-2026-10-24
[spec-threshold]: ../idea/transcribe-mvp-agent-plan.md#speaker-threshold-tuning-m5-2026-10-24
[spec-conditioning]: ../idea/transcribe-mvp-agent-plan.md#conditioning-on-previous-text-2026-10-24
[spec-short]: ../idea/transcribe-mvp-agent-plan.md#short-replies-2026-10-24
