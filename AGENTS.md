# AGENTS.md

This file guides AI coding agents and human contributors working in this
repository. It covers process: rules, commands and workflows. Product facts
(decisions, contracts, architecture) live in
[`docs/design.md`](docs/design.md); where the two differ on a product fact,
`docs/design.md` wins, and this file is updated to match.

## Project

`transcribe` is a CLI that turns a local audio file, in any language Whisper
supports (one per file), into a Markdown transcript with anonymous speaker
labels, using **local** AI models only: no data may leave the machine. It
targets older, CPU-only hardware, so prefer dependencies and models with
modest CPU and RAM needs and no GPU.

The MVP is implemented and benchmarked; one benchmark gate (speaker
attribution) still fails. Read before changing things:

- [`docs/design.md`](docs/design.md): scope,
  [decision log](docs/design.md#decision-log),
  [CLI contract](docs/design.md#cli-contract),
  [Markdown contract](docs/design.md#markdown-contract) and
  [architecture](docs/design.md#architecture).
- [`docs/how-it-works.md`](docs/how-it-works.md): each pipeline stage and the
  sources behind it.
- [`docs/benchmark.md`](docs/benchmark.md): method,
  [gates](docs/benchmark.md#gates) and [results](docs/benchmark.md#5-results).

## Hard rules

- **Ask the user first** before changing an engine, a model, the compute
  type, a `Settings` default, the CLI surface (flags, exit codes, output
  contract) or a benchmark gate. Never weaken a gate quietly. If
  `docs/design.md` does not cover a decision, ask.
- **Privacy:** never commit or upload audio, transcripts, model files or
  benchmark output that contains recognized text. Tests use synthetic or
  public-domain audio. `bench/` stays git-ignored.
- **No runtime network:** the CLI never downloads. A missing model exits
  with code 2 and names `scripts/provision_models.py`. Pass model directory
  paths, never hub names; `audio.offline_imports()` sets `HF_HUB_OFFLINE=1`.
- **Streams:** stdout carries only the Markdown document. Progress and
  errors go to stderr through `click.echo`. Error messages name the stage and
  the error type, never recognized text.
- **Pure modules:** `domain`, `languages`, `attribution`, `rendering` and
  `output` import neither Click nor engine packages.
- **Lazy engines:** `faster_whisper` and `sherpa_onnx` are imported inside
  adapter functions, so `--help` and the default tests never load them.
- **One seam for tunables:** the frozen `Settings` dataclass. The CLI uses
  its defaults, except `language` (`--language`) and `whisper_model_large`
  (`--quick`); add no public flag without asking.
- **No-clobber output:** publish files only through `output.publish_file`.
  Never use `os.replace` and never create directories.
- **Dependencies:** never build CTranslate2, ONNX Runtime or PyAV from
  source; if no wheel installs, stop and report. Keep the `av<19` and
  `onnxruntime==1.23.2` pins unless the user agrees.
- **Handover:** state what is implemented, tested, benchmarked and unknown.

## Architecture map

`cli.main` -> `application.run` -> preflight, decode, language detection, ASR,
diarization -> attribution -> rendering -> output. Details in
[`docs/design.md`](docs/design.md#architecture).

| Module           | Role                                                                  |
| ---------------- | --------------------------------------------------------------------- |
| `__main__.py`    | `python -m transcribe` calls `cli.main()`                             |
| `cli.py`         | Click command, exit-code wrapper                                      |
| `application.py` | Orchestration; engines injected through `Engines`                     |
| `settings.py`    | Frozen `Settings` dataclass, default models root                      |
| `errors.py`      | `TranscribeError` hierarchy with exit codes                           |
| `domain.py`      | `Word`, `Turn`, `Language`, `AttributedWord`, `Paragraph` (pure)      |
| `languages.py`   | Whisper language codes and names (pure)                               |
| `models.py`      | Manifest loading, model presence and hash checks                      |
| `audio.py`       | Decoding through faster-whisper and PyAV (lazy import)                |
| `asr.py`         | faster-whisper adapter: language detection, recognition (lazy import) |
| `diarization.py` | sherpa-onnx adapter (lazy import)                                     |
| `attribution.py` | Words to speakers, paragraphs (pure)                                  |
| `rendering.py`   | Markdown rendering and escaping (pure)                                |
| `progress.py`    | ASR and diarization progress on stderr: bar, lines or nothing         |
| `output.py`      | stdout and no-clobber file publication (pure)                         |
| `__init__.py`    | Package marker                                                        |

`scripts/` holds developer tools: `provision_models.py`, `benchmark.py`,
`play_reference.py` (benchmark reference text), `simulate_conversation.py`
(simulated conversations with an RTTM reference, for `--diarization-only`)
and `diarization_lab.py` (diarization experiments on cached ASR).

## Commands

The project uses `uv` (>= 0.13.5) and Python 3.13 (`src/` layout,
`uv_build` backend); 3.14 is excluded (see the decision log).

```sh
uv sync                          # install project + dev dependencies
uv run transcribe AUDIOFILE      # run the CLI (models must be provisioned)
uv run scripts/provision_models.py   # download and verify the models once
uv run ruff check                # lint
uv run ruff format               # format (CI runs `ruff format --check`)
uv run mypy                      # type check (strict: src/, tests/, scripts/)
uv run pytest                    # run default tests (no models, no network)
uv run pytest -m engine          # opt-in tests that need provisioned models
uv run pytest tests/test_cli.py::test_help_exits_zero   # run a single test
```

CI (`.github/workflows/check-python.yml`) runs `uv sync --locked`, so commit
`uv.lock` with every dependency change (`uv add <pkg>`, `uv add --dev <pkg>`).

## Models

`uv run transcribe`, `uv run pytest -m engine` and `scripts/benchmark.py`
read models from `~/Library/Application Support/transcribe/models/`
(`Settings.models_root`).

- `src/transcribe/model_manifest.json` is the single source of truth: name,
  license, source, revision, HTTPS URL and sha256 of every model file.
- `provision_models.py` downloads what is missing and verifies every hash.
  `--only NAME` limits it to one asset; `--models-root DIR` targets another
  directory (the benchmark then needs `--set models_root=DIR`).
- To add or change a model: set `sha256` to `null` in the manifest, run the
  script with `--record` on a trusted network, and commit the hashes.
- Runtime hash checks are off by default (`Settings.verify_model_hashes`).

## Changing the repo

- **Behaviour or feature:** update the tests, then `docs/design.md` (the
  contract, plus a dated decision-log entry if a decision changed),
  `docs/how-it-works.md` if a stage changed, and the README if users see it.
- **`Settings` default:** ask first. Tune on held-out audio, confirm on the
  evaluation recording (see `docs/benchmark.md`), add the dated table to
  `docs/benchmark.md`, a decision-log entry, the `Settings` docstring, and
  the README summary if headline numbers move.
- **Model:** ask first. Follow the manifest procedure above, update the
  `Settings` paths and the README model table, rerun `pytest -m engine` and
  the benchmark, and log the decision.
- **Dependency:** `uv add`, commit `uv.lock`; explain any pin in a comment
  in `pyproject.toml` and in the decision log.
- **Before finishing:** run `uv run ruff check`, `uv run ruff format
  --check`, `uv run mypy` and `uv run pytest`. Also run
  `uv run pytest -m engine` whenever an adapter, a `Settings` default or a
  dependency changed.

## Conventions enforced by tooling

- Ruff has a broad rule set (see `pyproject.toml`), with 120-char lines.
  Notable rules: `T20` forbids `print` (silence with a justified
  `# noqa: T201`), `PTH` requires `pathlib`, `FBT` forbids boolean positional
  args, `TC` moves type-only imports under `if TYPE_CHECKING:`, and isort
  wants two blank lines after imports.
- mypy runs in `strict` mode, so tests and scripts need full annotations
  too. Untyped engine packages (`faster_whisper`, `sherpa_onnx`) are exempted
  in `pyproject.toml`; keep their untyped surface inside the adapters.
- Python 3.13 evaluates annotations eagerly: modules that import types under
  `if TYPE_CHECKING:` need `from __future__ import annotations`.
- pytest runs with `filterwarnings = ["error"]` (any warning fails a test),
  `xfail_strict`, `--strict-markers` and `--import-mode=importlib`. Tests
  that need models carry the `engine` marker.
- Golden files under `tests/test_rendering/` come from `pytest-regressions`:
  regenerate them with `--force-regen` and never edit them by hand (they
  have their own markdownlint config).

## Markdown

All `*.md` files (except under `.claude/`) are linted in CI by
`markdownlint-cli2` (`.markdownlint-cli2.yaml`: 80-char lines outside tables
and code blocks, ATX headings, aligned tables) and link-checked by `lychee`.

Docs cite sources in APA 7 style, with an open-world tone (state what things
do; add a negation only where it prevents a likely misreading):

- cite inline where the claim appears, as `([Author, Year])`, a shortcut
  reference link; when the year is unknown, omit it (`([Author])`) in the
  citation and the reference, instead of writing "n.d.";
- end the doc with a `## References` section listing only cited sources,
  alphabetically, each as `[Author, Year] <APA 7 reference>`, followed by
  the link definitions `[Author, Year]: <url>`;
- link pinned versions (tags, DOIs) where they exist;
- keep benchmark numbers in `docs/benchmark.md` and the README summary;
  other docs link there instead of repeating them;
- date decisions and results; do not label them with milestone IDs.

## Commits

Use the repo's `caveman-commit` skill (`.claude/skills/caveman-commit/`) for
terse Conventional Commits messages, one logical change per commit.
