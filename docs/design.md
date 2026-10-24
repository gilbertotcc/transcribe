# Design

This document records what `transcribe` must do and why: its scope, the
decisions behind it, the contracts of the command line and of the Markdown
output, and the architecture rules that keep the code easy to change. How
each stage works is explained in [How it works](how-it-works.md); measured
results are in [Benchmark](benchmark.md).

## Goal and scope

`transcribe` turns one local English audio recording into a Markdown
transcript with anonymous speaker labels:

```sh
transcribe AUDIOFILE [--output OUTPUT.md]
```

- **Local only.** Models run on the machine that launches the command. The
  transcription run uses no network; models are provisioned once, ahead of
  time.
- **Older hardware first.** The reference host is a 2020 Intel MacBook Pro,
  CPU only ([host](benchmark.md#host)). Dependencies and models are chosen
  for modest CPU and RAM needs and no GPU.
- **Existing engines.** `faster-whisper` recognizes speech and `sherpa-onnx`
  finds speaker turns. The repository holds only the orchestration,
  attribution, formatting, output handling, tests and benchmark tooling: it
  trains no models and implements no recognition algorithm.
- **One command**, without subcommands or configuration files. Every tunable
  lives in `Settings` (D9).

## Decision log

Each decision states what was chosen and why. A change to a decision is a new
dated row, agreed with the maintainer; the numbers behind a decision live in
[Benchmark](benchmark.md).

| ID  | Date       | Decision                                                                                               | Rationale                                                                                                                                                                 |
| --- | ---------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | 2026-10-23 | Target a 2020 Intel MacBook Pro, CPU only, macOS 13 or later.                                          | The tool should run on hardware modern apps drop. No GPU code paths.                                                                                                      |
| D2  | 2026-10-23 | Python 3.13 (`requires-python = ">=3.13,<3.14"`); `onnxruntime==1.23.2` pinned on Intel Macs.          | `onnxruntime` 1.23.2 is the last release with Intel-Mac wheels, for CPython 3.10 to 3.13 on macOS 13+ ([Microsoft]). Python 3.14 has no such wheel.                       |
| D3  | 2026-10-23 | English only: Whisper `small.en`, `language="en"`, no language detection, no model flag.               | Keeps the model small and accurate for the target language. Non-English input is out of scope and not detected.                                                           |
| D4  | 2026-10-23 | Diarization is always on; labels are anonymous (`SPEAKER_01`, ..., `UNKNOWN`).                         | No names and no speaker-count flag: the number of speakers is inferred.                                                                                                   |
| D5  | 2026-10-23 | Decode in-process with PyAV, through `faster_whisper.decode_audio`.                                    | No `ffmpeg` binary, subprocess or temporary WAV. One 16 kHz mono float32 array (about 230 MB per hour) feeds both engines.                                                |
| D6  | 2026-10-23 | Run recognition, then diarization, sequentially in one process.                                        | Simplest design; peak memory stays within its gate ([gates](benchmark.md#gates)), so per-stage processes are unnecessary.                                                 |
| D7  | 2026-10-23 | Without `--output`, Markdown goes to stdout; diagnostics always go to stderr.                          | Allows `transcribe x.m4a > x.md` and pipes.                                                                                                                               |
| D8  | 2026-10-23 | Models are provisioned by `scripts/provision_models.py` and never downloaded at runtime.               | Keeps the transcription run offline. A missing model is a preflight error (exit 2) that names the script.                                                                 |
| D9  | 2026-10-23 | All tunables live in the frozen `Settings` dataclass; the CLI always uses the defaults.                | The benchmark varies settings (`--set FIELD=VALUE`) without adding public flags. This is the main extension point.                                                        |
| D10 | 2026-10-23 | The transcript is raw and not redacted.                                                                | README and the notice in every transcript say so.                                                                                                                         |
| D11 | 2026-10-23 | Load Whisper from its model directory path with `local_files_only=True`; set `HF_HUB_OFFLINE=1` first. | A model name could trigger a Hugging Face download; a path cannot.                                                                                                        |
| D12 | 2026-10-23 | Download the diarization models from the sherpa-onnx GitHub releases.                                  | The upstream pyannote repositories on Hugging Face are gated behind an account.                                                                                           |
| D13 | 2026-10-24 | Speaker embeddings: NeMo TitaNet small (English), over 3D-Speaker ERes2Net.                            | On a 240 s play excerpt it ran in less than half the time and over-clustered less.                                                                                        |
| D14 | 2026-10-24 | Pin `av>=18.1,<19`.                                                                                    | `faster-whisper` 1.2.1 passes `metadata_errors` to `av.open`, which PyAV 19 removed ([SYSTRAN, 2025]).                                                                    |
| D15 | 2026-10-24 | Memory gate relaxed from 4 GB to 4.5 GB peak RSS.                                                      | Agreed after the first 60 min measurements ([gates](benchmark.md#gates)).                                                                                                 |
| D16 | 2026-10-24 | `clustering_threshold` default 0.8 (the sherpa-onnx example uses 0.5).                                 | Tuned on held-out audio: fewer split speakers, while the two male voices of the 30 s test clip stay apart ([clustering threshold](benchmark.md#clustering-threshold)).    |
| D17 | 2026-10-24 | `condition_on_previous_text` default `False`.                                                          | `True` lowers WER slightly but more than doubles recognition time; headroom for older hardware matters more ([conditioning](benchmark.md#conditioning-on-previous-text)). |

## CLI contract

The command is a single Click `@click.command` with one argument and one
option besides `--help`. Implementation: `src/transcribe/cli.py`; tests:
`tests/test_cli.py`.

```sh
transcribe meeting.m4a
transcribe "recordings/meeting 01.m4a" --output transcript.md
transcribe --output transcript.md meeting.wav
transcribe -- -meeting.wav
python -m transcribe --help
```

### Input

- `AUDIOFILE` must exist, be readable, not be a directory and not be empty.
  Relative paths, spaces and Unicode work; symlinks to files are followed.
- The content is validated by decoding, not by extension. Any container that
  PyAV decodes works; the first audio stream is used. A file without an audio
  stream, or one that fails to decode, is a decode error (exit 1).
- Audio shorter than `Settings.min_audio_seconds` (0.5 s) skips both engines
  and yields the no-speech transcript.
- The source file is opened read-only.

### Output

| Situation                       | Behavior                                                 |
| ------------------------------- | -------------------------------------------------------- |
| No `--output`                   | The complete Markdown on stdout, written once at the end |
| `--output OUTPUT.md`            | A new UTF-8 file with mode 0600; nothing on stdout       |
| Destination exists (any kind)   | Refused, including dangling symlinks; no overwrite flag  |
| Extension not `.md` (any case)  | Refused (this also rejects `--output -`)                 |
| Parent directory missing        | Refused; directories are never created                   |
| Processing fails                | No transcript file is left behind                        |
| Stdout closed early (`\| head`) | No traceback; exit 1                                     |

The output path is validated before any model loads, so mistakes fail fast.
Publication (`output.publish_file`) then guarantees that nothing is
overwritten, even if a file appears while transcribing:

1. `tempfile.mkstemp` in the destination directory (a private 0600 file);
2. write, flush and `os.fsync`;
3. hard-link the temporary file to the destination, which raises
   `FileExistsError` if the destination exists
   ([Python Software Foundation]);
4. always unlink the temporary file.

`os.replace` is unsuitable because it overwrites, and `click.File("w")` is
unsuitable because Click would open, and could truncate, the file while
parsing arguments. A filesystem without hard links (for example exFAT) gets a
clear error asking for another directory.

### Exit codes

| Code | Meaning                                                         |
| ---- | --------------------------------------------------------------- |
| 0    | Success, or `--help`                                            |
| 1    | Decode, inference, attribution or output failure                |
| 2    | Usage error, unusable input or output path, missing model files |
| 130  | Interrupted (Ctrl-C)                                            |

Each `TranscribeError` subclass in `errors.py` carries its exit code. In
standalone mode Click reports Ctrl-C as "Aborted!" with exit 1, so the
console script `transcribe.cli:main` calls the command with
`standalone_mode=False` and maps the outcomes itself: `click.Abort` to 130,
`click.ClickException` to its own code, `TranscribeError` to its code
([Pallets]).

### Diagnostics

- stdout carries only the Markdown document.
- stderr shows one line when each stage (decode, ASR, diarization, render)
  starts and one with its elapsed seconds when it ends, plus periodic
  progress during ASR and diarization.
- Error messages name the stage and the exception type, never recognized
  text: `application` catches engine exceptions at the stage boundary and
  re-raises them as `TranscribeError` subclasses.
- Output goes through `click.echo`; `print` is forbidden by the linter.

## Markdown contract

Implementation: `src/transcribe/rendering.py` and `attribution.paragraphs`;
golden files: `tests/test_rendering/`.

```md
# Transcript: meeting.m4a

> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted.

[00:00:03] SPEAKER_01: Welcome to the meeting.

[00:00:08] SPEAKER_02: Let us begin.
```

- The title uses the input's basename, never its path.
- File names and recognized text render as plain prose: line breaks become
  spaces, and `` \ ` * _ [ ] < > | & # `` are backslash-escaped. Labels and
  timestamps are not escaped.
- Recognized wording is preserved: each word keeps the leading space
  faster-whisper gives it, words are concatenated verbatim and the paragraph
  is stripped.
- The timestamp is the paragraph start, rounded down, as `[hh:mm:ss]`; hours
  can exceed 99.
- Speakers are numbered by first appearance, with at least two digits
  (`SPEAKER_01`). Words that no speaker turn covers are `UNKNOWN`.
- A new paragraph starts on a speaker change, on a pause longer than
  `paragraph_gap_seconds` (2 s), or when a paragraph would exceed
  `paragraph_max_seconds` (30 s).
- Paragraphs are separated by one blank line, and the document ends with one
  newline.
- Without recognized speech, the document is the title, the notice and
  `No speech detected.`
- Rendering is a pure function: the same input gives byte-identical output.

## Architecture

```text
cli.main -> cli.cli (Click; input and output path checks)
  -> application.run
     -> output.validate_output_path
     -> application.build_transcript
        -> models.check_assets        preflight; exit 2 if a file is missing
        -> audio.decode               PyAV -> 16 kHz mono float32 array
        -> asr.transcribe             faster-whisper -> list[Word]
        -> diarization.diarize        sherpa-onnx -> list[Turn]
        -> attribution.attribute      -> list[AttributedWord]
        -> attribution.paragraphs     -> list[Paragraph]
        -> rendering.render_markdown  -> str
     -> output.write_stdout | output.publish_file
```

| Module                | Role                                                           |
| --------------------- | -------------------------------------------------------------- |
| `cli.py`              | Click command and the exit-code wrapper `main()`               |
| `__main__.py`         | `python -m transcribe`                                         |
| `application.py`      | Orchestration, stage timing, engine-error mapping; `Engines`   |
| `settings.py`         | Frozen `Settings` dataclass and the default models root        |
| `errors.py`           | `TranscribeError` hierarchy with exit codes                    |
| `domain.py`           | `Word`, `Turn`, `AttributedWord`, `Paragraph`; interval checks |
| `models.py`           | Manifest loading, required model paths, preflight, hash check  |
| `model_manifest.json` | Name, license, source, revision, URL and sha256 of every model |
| `audio.py`            | Decoding through PyAV                                          |
| `asr.py`              | faster-whisper adapter                                         |
| `diarization.py`      | sherpa-onnx adapter                                            |
| `attribution.py`      | Word-to-speaker attribution and paragraphs (pure)              |
| `rendering.py`        | Markdown rendering (pure)                                      |
| `output.py`           | Stdout and no-clobber file publication                         |

Outside the package, `scripts/provision_models.py` downloads and verifies the
models, `scripts/benchmark.py` runs the pipeline with `Settings` overrides
and writes JSON reports, and `scripts/play_reference.py` builds benchmark
reference texts ([Benchmark](benchmark.md)).

### Dependency rules

- `domain`, `attribution`, `rendering` and `output` import neither Click nor
  an engine package, so they stay pure and fast to test.
- Engine packages (`faster_whisper`, `av`, `sherpa_onnx`) are imported inside
  the adapter functions, and `cli` imports `application` only after argument
  parsing. `--help` and the default tests therefore never load an engine
  (`test_python_m_help_does_not_load_engines`).
- `application.build_transcript` and `run` take an optional `Engines` value
  (preflight, decoder, transcriber, diarizer as callables); `None` selects
  the real adapter. Tests pass fakes; there is no plugin registry.
- `Settings` is the only source of tunables. The CLI constructs it with
  defaults; tests and the benchmark override fields.
- Untyped engine APIs stay inside the adapters, which return domain records.

### Domain records

Frozen, slotted dataclasses. Times are seconds on the original recording,
before any VAD trimming.

- `Word(start, end, text)` and `Turn(start, end, speaker)` reject non-finite
  or negative times and `end < start`. `speaker` is the engine's opaque
  cluster ID.
- Adapters pass engine times through `domain.clamp_interval`, which clips
  them to the audio duration and repairs an `end` up to 1 ms before `start`.
- `AttributedWord(word, speaker, ambiguous)`: `speaker` is `None` when no
  turn covers the word; `ambiguous` marks a tie between speakers.
- `Paragraph(start, speaker_label, text)`.

The attribution rules (largest overlap, tie-breaking, zero-length words) are
described in [How it works](how-it-works.md#25-joining-words-and-speakers-attributionpy)
and covered by `tests/test_attribution.py`.

### Privacy rules

- The transcription run makes no network request (D8, D11).
- Benchmark reports contain measurements only: no recognized text and no
  file paths. Audio, transcripts and model files stay out of the repository;
  tests use synthetic or public-domain audio.
- The `Transcript` object holds recognized text, so it is never logged.

## References

[Microsoft] Microsoft. *onnxruntime 1.23.2* [Computer software]. PyPI.
Retrieved October 24, 2026, from
<https://pypi.org/project/onnxruntime/1.23.2/#files>

[Pallets] Pallets. *Exception handling and exit codes*. Click
documentation. Retrieved October 24, 2026, from
<https://click.palletsprojects.com/en/stable/exceptions/>

[Python Software Foundation] Python Software Foundation. *os:
Miscellaneous operating system interfaces*. Python 3.13 documentation.
Retrieved October 24, 2026, from
<https://docs.python.org/3.13/library/os.html#os.link>

[SYSTRAN, 2025] SYSTRAN. (2025). *faster-whisper* (Version 1.2.1) [Computer
software]. GitHub. <https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1>

[Microsoft]: https://pypi.org/project/onnxruntime/1.23.2/#files
[Pallets]: https://click.palletsprojects.com/en/stable/exceptions/
[Python Software Foundation]: https://docs.python.org/3.13/library/os.html#os.link
[SYSTRAN, 2025]: https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1
