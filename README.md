# transcribe

`transcribe` is a Python script that extracts text from audio files. It runs
locally using local AI models, so no data leaves the machine you launch it on.
It should work on *older* machines that are often not supported by modern
applications. It is vibe-coded, meaning I accept all the goods and sh*ts AI can
produce.

## Status

MVP, built to measure whether this approach is fast and accurate enough on a
2020 Intel MacBook Pro (CPU only). The design decisions and contracts are in
[Design](docs/design.md), the measured results in
[Benchmark](docs/benchmark.md).

Scope and limits:

- **One language per file, detected automatically.** Any language Whisper
  supports can be named with `--language`; English and Italian are the ones
  measured. Recordings that switch language, and translation, are not
  supported.
- **Speaker labels are automatic and anonymous** (`SPEAKER_01`, `SPEAKER_02`,
  …, `UNKNOWN`). They are inferred and can be wrong: the same person may get
  several labels.
- **Not redacted.** The transcript is raw and as confidential as the
  recording. Recording and transcribing people may require their consent.

## Install

Requirements: macOS 13 or later, Python 3.13 and
[uv](https://docs.astral.sh/uv/) 0.13.5 or later. No GPU is needed.

```sh
git clone https://github.com/gilbertotcc/transcribe.git
cd transcribe
uv sync
uv run scripts/provision_models.py
```

The last command downloads the pinned models (about 2.5 GB) once into
`~/Library/Application Support/transcribe/models/` and verifies their sha256
hashes. This is the only step that downloads models: `transcribe` itself never
downloads them and stops with exit code 2 if a model is missing.

Provisioning needs internet access to `huggingface.co` and `github.com`. It
is safe to rerun: models whose files already match their hashes are reported
as `up to date` and not downloaded again, and a file with a different hash
stops the run with a `sha256 mismatch` error. `--only NAME` provisions one
model (`faster-whisper-small`, `faster-whisper-large-v3-turbo`,
`faster-whisper-small.en`, `pyannote-segmentation-3.0` or
`nemo-titanet-small-en`, repeatable); `--quick` runs without
`faster-whisper-large-v3-turbo`. `transcribe` reads models only from the
default directory above.

| Model                                                             | Use                                            | License    |
| ----------------------------------------------------------------- | ---------------------------------------------- | ---------- |
| Whisper `small` (CTranslate2 conversion by Systran)               | language detection; recognition with `--quick` | MIT        |
| Whisper `large-v3-turbo` (CTranslate2 conversion, `dropbox-dash`) | speech recognition                             | MIT        |
| Whisper `small.en` (CTranslate2 conversion by Systran)            | English recognition with `--quick`             | MIT        |
| pyannote segmentation 3.0 (ONNX export by sherpa-onnx)            | speaker turns                                  | MIT        |
| NVIDIA NeMo TitaNet small (ONNX export by sherpa-onnx)            | speaker embeddings                             | Apache-2.0 |

## Usage

```sh
uv run transcribe meeting.m4a                         # Markdown on stdout
uv run transcribe meeting.m4a --output meeting.md     # or into a new file
uv run transcribe riunione.m4a --language it          # skip detection
uv run transcribe meeting.m4a --quick                 # faster, less accurate
uv run transcribe -- -odd-name.wav                    # names starting with -
```

The language is detected from the first speech in the recording and named
on stderr; a warning suggests `--language` when detection is unsure.
`--quick` transcribes with Whisper `small` instead of `large-v3-turbo`.

Any format PyAV can decode works (WAV, MP3, M4A, …). Progress goes to stderr.
`--output` must end in `.md`, its directory must exist, and an existing file
is never overwritten. Exit codes: 0 success, 1 processing failure, 2 usage or
missing models, 130 interrupted.

The output looks like this:

```md
# Transcript: meeting.m4a

Language: English (en), detected

> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted.

[00:00:03] SPEAKER_01: Welcome to the meeting.

[00:00:08] SPEAKER_02: Let us begin.
```

## Benchmark summary

Measured on a 2020 13" MacBook Pro (Intel Core i5-1038NG7, 4 cores,
16 GB RAM, macOS 26.7) with a 60 min public-domain LibriVox play reading
(7 readers, English) and default settings:

| Measure                                      | Result            | Target |
| -------------------------------------------- | ----------------- | ------ |
| Processing time (ASR 21 min, speakers 8 min) | 29 min (RTF 0.48) | 60 min |
| Peak memory                                  | 4.6 GB            | 6 GB   |
| Word error rate                              | 5.1%              | 8%     |
| Words given the right speaker label          | 74%               | 90%    |

With `--quick` the same recording took 21 min (RTF 0.35) and 3.6 GB, with
6.4% WER. On simulated Italian conversations of read speech, WER was 19-22%
by default and 24-28% with `--quick`; those references spell numbers out,
so these values overstate the errors.

Speaker labels are the weak spot: one voice often gets several labels
(40 labels for 7 readers in the 60 min test), although each label rarely
mixes two voices. The test material, procedure, threshold sweep and other
experiments are in [Benchmark](docs/benchmark.md#5-results).

## Documentation

| Document                             | Contents                                                        |
| ------------------------------------ | --------------------------------------------------------------- |
| [Design](docs/design.md)             | Scope, decision log, CLI and Markdown contracts, architecture   |
| [How it works](docs/how-it-works.md) | Each stage from waveform to Markdown, with its sources          |
| [Benchmark](docs/benchmark.md)       | Test material, procedure, gates and all measured results        |
| [`AGENTS.md`](AGENTS.md)             | Contributor guide: rules, commands, conventions, change recipes |

## Development

See [`AGENTS.md`](AGENTS.md) for commands and conventions. It is written for
AI coding agents and human contributors alike. Benchmarks run with
`uv run scripts/benchmark.py AUDIO --label NAME [--reference REF]` and write
JSON reports (measurements only, no text) to the git-ignored `bench/`
directory.

## License

[MIT](LICENSE).
