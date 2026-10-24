# transcribe

`transcribe` is a Python script that extracts text from audio files. It runs
locally using local AI models, so no data leaves the machine you launch it on.
It should work on *older* machines that are often not supported by modern
applications. It is vibe-coded, meaning I accept all the goods and sh*ts AI can
produce.

## Status

MVP, built to measure whether this approach is fast and accurate enough on a
2020 Intel MacBook Pro (CPU only). The specification and the measured results
are in
[`idea/transcribe-mvp-agent-plan.md`](idea/transcribe-mvp-agent-plan.md).
AI coding agents should start from [`AGENTS.md`](AGENTS.md).

Scope and limits:

- **English only.** Other languages are neither supported nor detected.
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

The last command downloads the pinned models (about 510 MB) once into
`~/Library/Application Support/transcribe/models/` and verifies their sha256
hashes. This is the only step that downloads models: `transcribe` itself never
downloads them and stops with exit code 2 if a model is missing.

Provisioning needs internet access to `huggingface.co` and `github.com`. It
is safe to rerun: models whose files already match their hashes are reported
as `up to date` and not downloaded again, and a file with a different hash
stops the run with a `sha256 mismatch` error. `--only NAME` provisions one
model (`faster-whisper-small.en`, `pyannote-segmentation-3.0` or
`nemo-titanet-small-en`). `transcribe` reads models only from the default
directory above.

| Model                                                         | Use                 | License    |
| ------------------------------------------------------------- | ------------------- | ---------- |
| Whisper `small.en` (CTranslate2 conversion by Systran)        | speech recognition  | MIT        |
| pyannote segmentation 3.0 (ONNX export by sherpa-onnx)        | speaker turns       | MIT        |
| NVIDIA NeMo TitaNet small (ONNX export by sherpa-onnx)        | speaker embeddings  | Apache-2.0 |

## Usage

```sh
uv run transcribe meeting.m4a                         # Markdown on stdout
uv run transcribe meeting.m4a --output meeting.md     # or into a new file
uv run transcribe -- -odd-name.wav                    # names starting with -
```

Any format PyAV can decode works (WAV, MP3, M4A, …). Progress goes to stderr.
`--output` must end in `.md`, its directory must exist, and an existing file
is never overwritten. Exit codes: 0 success, 1 processing failure, 2 usage or
missing models, 130 interrupted.

The output looks like this:

```md
# Transcript: meeting.m4a

> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted.

[00:00:03] SPEAKER_01: Welcome to the meeting.

[00:00:08] SPEAKER_02: Let us begin.
```

## How it works

[`docs/how-it-works.md`](docs/how-it-works.md) explains each stage, from
waveform to recognized words and speaker turns to Markdown, what the benchmark
numbers measure, and the sources behind each stage.

## Benchmark summary

Measured on a 2020 13" MacBook Pro (Intel Core i5-1038NG7, 4 cores,
16 GB RAM, macOS 26.7) with a 60 min public-domain LibriVox play reading
(7 readers) and default settings:

| Measure                                      | Result            | Target |
| -------------------------------------------- | ----------------- | ------ |
| Processing time (ASR 11 min, speakers 7 min) | 18 min (RTF 0.30) | 60 min |
| Peak memory                                  | 3.4 GB            | 4.5 GB |
| Word error rate                              | 6.3%              | 8%     |
| Words given the right speaker label          | 74%               | 90%    |

Speaker labels are the weak spot: one voice often gets several labels
(40 labels for 7 readers in the 60 min test), although each label rarely
mixes two voices. Details and the threshold sweep are in section 9 of the
[specification](idea/transcribe-mvp-agent-plan.md); the test material,
procedure and findings are described in [Benchmark](docs/benchmark.md).

## Development

See [`AGENTS.md`](AGENTS.md) for commands and conventions. Benchmarks run with
`uv run scripts/benchmark.py AUDIO --label NAME [--reference REF]` and write
JSON reports (measurements only, no text) to the git-ignored `bench/`
directory.
