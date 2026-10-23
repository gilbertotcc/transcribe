# transcribe: MVP specification and implementation plan

Updated: 2026-10-23.
Status: reviewed, ready for implementation. Feasibility and performance on the
target machine are still unmeasured.

## 0. How to use this document

This is the brief for the coding agent that implements the MVP on the
**target machine**. Read `AGENTS.md` first; it remains authoritative for
tooling, lint and commit conventions. Then follow the milestones in section 10
in order.

- Ask the user when a decision is not covered here, or when a stop gate fails.
  Do not silently change requirements, swap engines or weaken a gate.
- Commit at the end of each milestone (use the `caveman-commit` skill).
- Never commit or upload audio, transcripts, model files or benchmark output
  that contains recognized text. Use synthetic or public audio in tests.
- Handover must state what is implemented, tested, benchmarked and unknown.

## 1. Goal and scope

Build one command that turns a local English audio recording into a Markdown
transcript with anonymous speaker labels, using local models only:

```sh
transcribe AUDIOFILE [--output OUTPUT.md]
```

The MVP exists to **measure** whether this approach is fast and accurate
enough on an older CPU-only laptop. Its scope is therefore the working
pipeline plus the instrumentation needed to benchmark it. Hardening beyond the
essentials is deferred (section 12), but the module boundaries in section 5
are designed so it can be added without restructuring.

Use existing engines: `faster-whisper` for speech recognition and
`sherpa-onnx` for speaker diarization. Write only the orchestration,
attribution, formatting, output handling, tests and benchmark tooling. Do not
train models, write recognition algorithms or add an agent framework.

## 2. Decisions and assumptions

| ID  | Decision                                                                                | Rationale / consequence                                                                                                                                              |
| --- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Target host: 2020 Intel MacBook Pro, CPU only. Exact CPU, RAM and macOS recorded in M0. | No GPU code paths. Stop if macOS is older than 13 (see D2).                                                                                                          |
| D2  | **Python 3.13**, not the repo's current 3.14.                                           | `faster-whisper` requires `onnxruntime`, which has no Intel-Mac wheel for 3.14. Last Intel-Mac release is 1.23.2 (cp310–cp313, macOS 13+). Pin it for that platform. |
| D3  | English only: model `small.en`, no language detection, no model flag.                   | Non-English input is out of scope and not detected.                                                                                                                  |
| D4  | Mandatory anonymous diarization: `SPEAKER_01`, `SPEAKER_02`, …, plus `UNKNOWN`.         | No names, no speaker-count flag.                                                                                                                                     |
| D5  | Decode audio in-process with PyAV (already a `faster-whisper` dependency).              | No external `ffmpeg` binary, no subprocess, no temporary WAV. One 16 kHz mono float32 array (~230 MB per hour) feeds both engines.                                   |
| D6  | Run ASR then diarization sequentially, in-process.                                      | Process isolation only if measured memory demands it (deferred).                                                                                                     |
| D7  | Omitted `--output` writes Markdown to stdout; diagnostics always go to stderr.          | Allows `transcribe x.m4a > x.md` and pipes.                                                                                                                          |
| D8  | Models are provisioned once by a developer script, never downloaded at runtime.         | Missing models are a preflight error (exit 2) that names the provisioning script.                                                                                    |
| D9  | All tunables live in one frozen `Settings` dataclass; the CLI uses defaults only.       | Benchmark script can vary settings without adding public flags. This is the main extension seam.                                                                     |
| D10 | Output is confidential raw transcription, not redacted.                                 | README and the Markdown notice say so.                                                                                                                               |

### Measured host facts (M0, 2026-10-24)

- CPU: Intel Core i5-1038NG7 @ 2.00 GHz, 4 cores / 8 threads, x86_64.
- RAM: 16 GB. macOS 26.7.1 (25G241), so the macOS 13 floor in D1 is met.
- Python 3.13.7, uv 0.13.5. All engine packages install as binary wheels:
  `ctranslate2` 4.8.3, `av` 19.0.2, `onnxruntime` 1.23.2, `sherpa-onnx` 2.0.0,
  `tokenizers` 0.23.3, `faster-whisper` 1.2.1.

## 3. CLI contract

Implement with Click: a single `@click.command`, no group, no subcommands, no
flags other than `--output` and the standard `--help`.

```sh
transcribe meeting.m4a
transcribe "recordings/meeting 01.m4a" --output transcript.md
transcribe --output transcript.md meeting.wav
transcribe -- ./-meeting.wav
transcribe --help
```

### Input

- One existing, readable, non-empty regular file (symlinks to regular files
  are fine). Relative paths, spaces and Unicode must work.
- Content is validated by decoding, not by extension. WAV, MP3 and M4A must
  work. The first audio stream is used.
- The source file is never modified.

### Output

| Situation                      | Behavior                                                   |
| ------------------------------ | ---------------------------------------------------------- |
| No `--output`                  | Complete Markdown on stdout, written once after rendering. |
| `--output OUTPUT.md`           | UTF-8 file, nothing on stdout.                             |
| Destination exists (any kind)  | Refuse. No overwrite flag, no alternative name.            |
| Extension not `.md` (any case) | Refuse (this also rejects `--output -`).                   |
| Parent directory missing       | Refuse; never create directories.                          |
| Processing fails               | No transcript file is left behind.                         |
| Stdout pipe closed (`\| head`) | No traceback; exit 1.                                      |

Validate the output path **before** any inference (fail fast), and again
implicitly at publication. Publication must not overwrite. Use this approach:
`tempfile.mkstemp` in the destination directory, write, flush, `os.fsync`,
`os.link(tmp, destination)` (raises `FileExistsError` if something appeared
meanwhile), and always unlink the temporary file. The resulting file has mode
0600, which suits a confidential transcript. `os.replace` is not acceptable
because it overwrites. If `os.link` is unsupported on the destination
filesystem, fail with a clear message (fallback is deferred).

### Exit codes

| Code | Meaning                                                       |
| ---- | ------------------------------------------------------------- |
| 0    | Success, or `--help`.                                         |
| 1    | Decode, inference, attribution or output failure.             |
| 2    | Usage error, invalid input/output path, missing model assets. |
| 130  | Interrupted (Ctrl-C).                                         |

Click's standalone mode turns Ctrl-C into "Aborted!" with exit 1, so the
console-script entry point must be a small wrapper that calls the command with
`standalone_mode=False` and maps the outcomes. Illustrative; verify the
details against the pinned Click version:

```python
@click.command(name="transcribe")
@click.argument(
    "audio_file",
    metavar="AUDIOFILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    metavar="OUTPUT.md",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write Markdown to a new .md file instead of stdout.",
)
def cli(audio_file: Path, output_path: Path | None) -> None:
    """Transcribe one local English audio file with anonymous speaker labels."""
    from transcribe.application import run  # heavy imports stay off --help

    run(audio_file, output_path, Settings())


def main() -> None:
    try:
        code = cli.main(standalone_mode=False)  # returns exit code on --help
    except click.Abort:  # Click converts KeyboardInterrupt to Abort
        sys.exit(130)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except TranscribeError as exc:
        click.echo(f"transcribe: {exc}", err=True)
        sys.exit(exc.exit_code)
    sys.exit(code if isinstance(code, int) else 0)
```

Do not use `click.File("w")` for the output: Click would open (and could
truncate) it during parsing.

### Diagnostics

- stderr shows short progress lines per stage with elapsed seconds (decode,
  ASR, diarization, render). This doubles as coarse benchmark data.
- Error messages name the stage and the error type, never recognized text.
  Engine exceptions are caught at the application boundary and re-raised as
  `TranscribeError` subclasses.
- stdout carries only the Markdown document. Engines must not print to it;
  verify this in the engine tests.
- Use `click.echo` (ruff `T20` forbids `print`).

## 4. Markdown contract

```md
# Transcript: meeting.m4a

> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted.

[00:00:03] SPEAKER_01: Welcome to the meeting.

[00:00:08] SPEAKER_02: Let us begin.
```

- Title uses the input basename only, never the full path.
- Escape filenames and recognized text so they render as plain prose:
  backslash-escape `` \ ` * _ [ ] < > | & # `` in text (labels and timestamps
  are not escaped).
- Preserve recognized wording. `faster-whisper` words carry their own leading
  space (`" Hello"`): concatenate words verbatim, then strip the paragraph.
- Timestamp is `floor(paragraph start)` as `[hh:mm:ss]`; hours may exceed 99.
- Speaker labels are numbered by first appearance in the transcript,
  two digits minimum (`SPEAKER_01`).
- Paragraphs are separated by a blank line; the file ends with one newline.
- No recognized speech: title, notice, then `No speech detected.`
- Rendering is a pure function: same input gives byte-identical output.

## 5. Architecture

```text
cli (Click, path checks) -> application.run
  -> models.check_assets            (preflight; exit 2 if missing)
  -> audio.decode                   (PyAV -> float32 mono 16 kHz array)
  -> asr.transcribe                 (faster-whisper -> list[Word])
  -> diarization.diarize            (sherpa-onnx -> list[Turn])
  -> attribution.attribute          (-> list[AttributedWord])
  -> attribution.paragraphs         (-> list[Paragraph])
  -> rendering.render_markdown      (-> str)
  -> output.write_stdout | output.publish_file
```

Layout (extends the existing `src/transcribe/` package):

```text
src/transcribe/
    __init__.py         # package marker; no CLI code
    __main__.py         # python -m transcribe -> cli.main()
    cli.py              # Click command and exit-code wrapper
    application.py      # orchestration; engines injected for tests
    settings.py         # frozen Settings dataclass, default model paths
    errors.py           # TranscribeError hierarchy with exit codes
    domain.py           # Word, Turn, AttributedWord, Paragraph
    audio.py            # decode + basic validation
    asr.py              # faster-whisper adapter (lazy import)
    diarization.py      # sherpa-onnx adapter (lazy import)
    attribution.py      # word-to-speaker, paragraphs (pure)
    rendering.py        # Markdown (pure)
    output.py           # stdout and no-clobber file publication
    models.py           # asset paths, presence/hash check
    model_manifest.json # relative paths, URLs, revisions, sha256
scripts/
    provision_models.py # developer setup: download + verify models
    benchmark.py        # run pipeline with timing/memory; JSON report
tests/
```

Rules:

- Update `[project.scripts]` to `transcribe = "transcribe.cli:main"`.
- `domain`, `attribution`, `rendering` and `output` must not import Click or
  engine packages.
- `application.run` takes the transcriber and diarizer as parameters (simple
  callables or `typing.Protocol`), with the real adapters as defaults. Tests
  pass fakes. No plugin registry.
- Engine packages are imported inside the adapter functions, so `--help` and
  the pure tests never load them.

### Domain records

Frozen, slotted dataclasses. Times are seconds on the original, untrimmed
recording. `__post_init__` rejects NaN/Inf, negative times and `end < start`
(allow a 1 ms tolerance and clamp to the audio duration).

- `Word(start, end, text)`
- `Turn(start, end, speaker)` where `speaker` is the engine's opaque integer
- `AttributedWord(word, speaker: int | None, ambiguous: bool)`
- `Paragraph(start, speaker_label, text)`

### Attribution (pure, fully unit-tested)

For each word:

1. Merge overlapping turns of the same speaker first (no double counting).
2. Choose the speaker with the largest positive overlap with the word.
3. Break ties by the speaker's earliest turn start, then by speaker ID; set
   `ambiguous=True`.
4. No overlap: `UNKNOWN` (do not carry over the previous speaker).
5. Zero-duration word: the speaker whose turn contains its time, same tie
   rule; otherwise `UNKNOWN`.

Paragraphs: group consecutive words with the same label; start a new
paragraph on a speaker change, a gap over 2.0 s between words, or when the
paragraph would exceed 30 s. Both thresholds live in `Settings`.

## 6. Engines and models

### Decoding

Use `faster_whisper.decode_audio(file_object, sampling_rate=16000)`, passing a
file object opened by our code (`open(path, "rb")`). It decodes the first
audio stream to mono float32. Map PyAV errors to a decode error (exit 1).
Reject audio with no samples or shorter than 0.5 s as `No speech detected`
output rather than an error. Container-level restrictions on remote
references are deferred (section 12).

### ASR baseline

`faster_whisper.WhisperModel(<local model dir>, device="cpu",
compute_type="int8", cpu_threads=4, local_files_only=True)` then
`model.transcribe(audio, language="en", beam_size=5,
condition_on_previous_text=False, word_timestamps=True, vad_filter=True)`.

- The returned segments are a lazy generator; consume it fully.
- Word timestamps are in original-recording time even with VAD; verify with a
  test recording that has leading silence.
- Pass the model **directory path**, never the name `small.en`, so nothing
  can trigger a download. Set `HF_HUB_OFFLINE=1` before importing as a belt
  and braces measure.
- Model: CTranslate2 conversion of Whisper `small.en`
  (`Systran/faster-whisper-small.en` on Hugging Face), pinned to a commit.

### Diarization baseline

Follow the official sherpa-onnx offline diarization example:
`OfflineSpeakerDiarization` with a pyannote segmentation 3.0 ONNX model and a
speaker-embedding ONNX model, `FastClusteringConfig(num_clusters=-1,
threshold=...)`, then `process(audio).sort_by_start_time()`.

- Download models from the sherpa-onnx GitHub releases, not gated
  Hugging Face repos.
- Embedding candidates: NeMo TitaNet small (English) and 3D-Speaker ERes2Net.
  Pick one in M1 on a real recording and record why.
- Start with the example's clustering threshold and duration parameters, and
  keep them in `Settings`. Tune only on recordings not used for the final
  evaluation.
- Check that `sd.sample_rate == 16000`.

### Model provisioning

`scripts/provision_models.py` (developer tool, not part of the installed CLI)
downloads the pinned assets into the models root, verifies sha256 against
`model_manifest.json`, and prints a summary. The first run on the target
machine records the hashes and licenses into the manifest, which is then
committed.

- Default models root: `~/Library/Application Support/transcribe/models/`
  (defined in `settings.py`, overridable only through `Settings`, for tests
  and benchmarks).
- Record each asset's source URL, revision, sha256 and license in the
  manifest. Stop and ask if a license is unclear or access requires an
  account or token.
- At runtime `models.check_assets` verifies that required files exist.
  Hashing large files at every run is optional (benchmark it).
- Models are never stored in the repository.

## 7. Dependencies and tooling changes

Repository changes (milestone M0):

- `.python-version` -> `3.13`; `requires-python = ">=3.13,<3.14"`; update
  `AGENTS.md` (Python version, project status, new modules) accordingly.
- Runtime dependencies: `click`, `faster-whisper`, `sherpa-onnx`, `numpy`
  (used directly). Add `onnxruntime==1.23.2` with a marker for
  `sys_platform == 'darwin' and platform_machine == 'x86_64'`, so the lock
  resolves to a wheel that exists for the target.
- Pin with `uv add`, commit `uv.lock`. CI (`ubuntu-latest`) must still pass
  `uv sync --locked`.
- Do not build CTranslate2, ONNX Runtime or PyAV from source. If no wheel
  installs on the target, stop and report.
- mypy strict: add `[[tool.mypy.overrides]]` with `ignore_missing_imports`
  only for engine modules that ship no type information
  (likely `faster_whisper`, `sherpa_onnx`). Keep the untyped surface inside
  the adapters.
- pytest has `filterwarnings = ["error"]`: if engine imports emit third-party
  warnings, add narrow, commented ignores; never a blanket ignore.
- Register an `engine` marker (required by `--strict-markers`) and exclude it
  by default (for example `addopts` gains `-m "not engine"`).
- Remove the placeholder greeting and update `tests/test_smoke.py`.

## 8. Testing

Default suite (CI, no models, no network):

- CLI via `click.testing.CliRunner`: `--help`, missing input, extra
  arguments, unknown option, directory input, empty file, `--` with a
  dash-prefixed name, output checks (exists, dangling symlink, wrong
  extension, `-`, missing parent). Assert `result.stdout` and
  `result.stderr` separately, and assert exit codes.
- One subprocess test: `python -m transcribe --help` exits 0 and does not
  import engine modules (check `sys.modules` in a child, or time-box it).
- Application with fake engines: stdout mode, file mode, no speech,
  `UNKNOWN` words, engine exception gives exit 1 with no recognized text in
  stderr and no output file, `KeyboardInterrupt` from a fake gives 130.
- Pure units: domain validation, attribution (ties, merges, zero-duration,
  no turns), paragraphs (gap, 30 s, speaker change), timestamps (> 24 h),
  escaping, rendering golden files via `pytest-regressions`.
- Output: publication success, refusal on existing file, temporary file
  removed on failure.
- Decoding: generate a short WAV with the stdlib `wave` module in `tmp_path`
  and decode it with PyAV; a corrupt file maps to exit 1.

Opt-in engine tests (`pytest -m engine`, skipped when models are absent):
a short public-domain English clip with two speakers; check that words have
plausible, monotonically increasing timestamps, at least two speakers are
found, and stdout contains only Markdown.

## 9. Benchmark

`scripts/benchmark.py AUDIO --label NAME [--reference REF.txt]` runs the
pipeline through `application` with `Settings` overrides and writes a JSON
report. It contains no recognized text and no file path, only the label:

- Host: CPU model (`sysctl -n machdep.cpu.brand_string`), core count, RAM,
  macOS version, Python version, versions of the engine packages, model
  hashes, all `Settings` values.
- Audio duration; seconds per stage and total; real-time factor.
- Peak RSS (`resource.getrusage`; macOS reports bytes, Linux kibibytes,
  so normalize).
- Speakers found; share of words labeled `UNKNOWN`; ambiguous share.
- WER, when a hand-checked reference text exists (add `jiwer` as a dev
  dependency, or a small word-level edit distance).

Run it on a 30 s clip, a representative 10 min recording and a full 60 min
recording, cold and warm. Keep reports and recordings outside the repository,
or in a git-ignored directory.

Provisional gates (confirm with the user after the first measurements; never
relax silently):

| Gate                | Target                                                     |
| ------------------- | ---------------------------------------------------------- |
| Runtime             | 60 min of audio in at most 60 min (RTF <= 1.0)             |
| Memory              | Peak RSS <= 4 GB (agreed 2026-10-24; host has 16 GB)       |
| WER                 | <= 8% on a hand-checked reference                          |
| Speaker attribution | >= 90% of words correctly attributed (manual or reference) |

If a gate fails, try these as measured experiments, one at a time, and report
each result: `cpu_threads` variations; `BatchedInferencePipeline` with batch
size 2 and 4; `base.en` instead of `small.en`. If ASR is still too slow, report
and propose whisper.cpp; do not implement a second backend without approval.

## 10. Milestones

| ID  | Milestone         | Deliverables                                                                                     | Stop gate                                                                                 |
| --- | ----------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| M0  | Environment       | Host facts recorded; Python 3.13 switch; dependencies added and locked; `AGENTS.md` updated.     | `uv sync` installs binary wheels on the target, and CI-equivalent checks pass. Else stop. |
| M1  | Feasibility spike | Provisioning script and manifest; throwaway check that both engines run on a 30 s clip offline.  | Both engines produce output with networking off. Else stop and report.                    |
| M2  | CLI and output    | `cli.py`, `errors.py`, `settings.py`, `output.py`, fake `application`, CLI and output tests.     | `transcribe --help` works without loading engines; all default tests pass.                |
| M3  | Pure core         | `domain.py`, `attribution.py`, `rendering.py`, golden tests.                                     | All attribution and rendering cases in section 8 pass.                                    |
| M4  | Pipeline          | `audio.py`, `asr.py`, `diarization.py`, `models.py`, real `application.run`, engine tests.       | End-to-end Markdown for WAV, MP3 and M4A; leading-silence timestamps correct.             |
| M5  | Benchmark         | `benchmark.py`; reports for 30 s, 10 min and 60 min recordings; summary shared with the user.    | Each gate is reported as passed, failed or unknown. Discuss failures before optimizing.   |
| M6  | Handover          | README: install, provisioning, usage, scope limits, benchmark summary; deferred backlog updated. | All checks in `AGENTS.md` pass.                                                           |

## 11. Definition of done

- [ ] Exactly `transcribe AUDIOFILE [--output OUTPUT.md]`, built with Click.
- [ ] Python 3.13, dependencies locked, CI green.
- [ ] `uv run transcribe` works from the repository; installed-tool workflow
      documented in README (section 12 lists hardening of it).
- [ ] Single ASR pass plus mandatory anonymous diarization; `UNKNOWN` shown.
- [ ] Stdout/stderr separation, no-clobber file output and exit codes tested.
- [ ] No runtime downloads; models provisioned and pinned with hashes and
      licenses.
- [ ] Default tests need no models or network; engine tests are opt-in.
- [ ] Benchmark reports show measured results and unmet gates.
- [ ] README states: English only, automatic speaker labels, no redaction,
      interrupted runs restart from scratch.

## 12. Deferred backlog

Not part of the MVP. Each item names where it plugs in.

- **Input hardening** (`cli.py`, `audio.py`): explicit FIFO/device
  rejection, hard-link and symlink alias detection against the output,
  restricting PyAV/FFmpeg protocols so containers cannot reference remote
  resources.
- **Output hardening** (`output.py`): fallback for filesystems without hard
  links (for example exFAT), concurrent-creator race tests, disk-full
  injection.
- **Isolation and privacy** (`application.py`): per-stage worker processes
  if memory requires it; outbound-network-denied test run; audit of library
  cache writes; restrictive cleanup of orphaned temporary files.
- **Installation**: `uv tool install` with constraints exported from
  `uv.lock`, smoke-tested outside the repository.
- **Features**: speaker naming, redaction, other languages, model selection,
  second-pass recognition, checkpoints and resume, configuration files.
- **Governance**: recording and transcribing people may be subject to consent
  and data-protection rules; review before using the tool on others'
  recordings.

## 13. References

Consulted 2026-10-23. Documentation is rolling: verify against the pinned
versions.

- Click: [single commands](https://click.palletsprojects.com/en/stable/quickstart/),
  [arguments](https://click.palletsprojects.com/en/stable/arguments/),
  [exceptions and exit codes](https://click.palletsprojects.com/en/stable/exceptions/),
  [testing](https://click.palletsprojects.com/en/stable/testing/).
- faster-whisper: [README](https://github.com/SYSTRAN/faster-whisper),
  [transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py),
  [audio.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/audio.py),
  [small.en model](https://huggingface.co/Systran/faster-whisper-small.en).
- sherpa-onnx: [diarization guide](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html),
  [official Python example](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py),
  [segmentation models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models),
  [embedding models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
  (the tag name is misspelled upstream).
- [onnxruntime on PyPI](https://pypi.org/project/onnxruntime/1.23.2/#files):
  last release with Intel-Mac wheels.
- [PyAV](https://pyav.basswood-io.com/docs/stable/).
- Python: [os.link](https://docs.python.org/3.13/library/os.html#os.link),
  [tempfile](https://docs.python.org/3.13/library/tempfile.html),
  [resource](https://docs.python.org/3.13/library/resource.html).
- Testing: [pytest](https://docs.pytest.org/en/stable/),
  [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/).
- [uv tools](https://docs.astral.sh/uv/guides/tools/).
- Fallback only: [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
