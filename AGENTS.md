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
modest CPU and RAM needs; the defaults need no GPU. On newer hosts a
hardware profile (`config/hardware.py`) may use the GPU, if the benchmark
shows it does at least as well as the 2020 Intel reference.

The MVP is implemented and benchmarked; it meets all four benchmark gates
on the 60 min evaluation recording. Read before changing things:

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
  paths, never hub names; `pipeline.offline.offline_imports()` sets
  `HF_HUB_OFFLINE=1`.
- **Streams:** stdout carries only the Markdown document. Progress and
  errors go to stderr through `click.echo` in `cli/`. Error messages name
  the stage and the error type, never recognized text.
- **Layers:** only `cli/` imports Click; the pipeline reports through an
  `Observer` (`pipeline/events.py`). Imports point `cli` -> `pipeline` ->
  `pipeline.steps`; `config`, `models`, `domain` and `languages` never
  import `pipeline` or `cli`.
- **Pure modules:** `domain`, `languages`, `steps.attribution`,
  `steps.diarization.clustering`, `steps.rendering` and `steps.output`
  import neither Click nor engine packages.
- **Lazy engines:** `faster_whisper`, `sherpa_onnx` and `mlx_whisper` are
  imported inside adapter functions, so `--help` and the default tests never
  load them.
- **One seam for tunables:** the frozen `Settings` dataclass. The CLI uses
  its defaults plus the host's profile
  (`config.hardware.PROFILE_OVERRIDES`), except `language` (`--language`)
  and `whisper_model_large` (`--quick`); add no public flag without asking.
  A profile override is a `Settings` default for that host: ask first.
- **No-clobber output:** publish files only through
  `steps.output.publish_file`.
  Never use `os.replace` and never create directories.
- **Dependencies:** never build CTranslate2, ONNX Runtime or PyAV from
  source; if no wheel installs, stop and report. Keep the `av<19` and
  `onnxruntime==1.23.2` pins unless the user agrees.
- **Handover:** state what is implemented, tested, benchmarked and unknown.

## Architecture map

`cli.main` -> `config.hardware` profile -> `pipeline.run` -> preflight,
then the `pipeline/steps/` in order: decode, language, ASR, diarization
(with re-clustering), attribution, rendering, output. The package layout and
engine adapters are in [`docs/design.md`](docs/design.md#architecture).
Tests mirror the package under `tests/`; `tests/test_architecture.py` checks
the import rules.

`scripts/` holds developer tools: `provision_models.py`, `benchmark.py`,
`fetch_bench_material.py` (downloads and rebuilds `bench/` from
`bench_material.json`), `play_reference.py` (benchmark reference text),
`simulate_conversation.py`
(simulated conversations with an RTTM reference, for `--diarization-only`),
`diarization_lab.py` (diarization experiments on cached ASR) and
`token_report.py` (Claude Code token usage, from local session logs).

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
uv run pytest tests/cli/test_cli.py::test_help_exits_zero   # run a single test
```

CI (`.github/workflows/check-python.yml`) runs `uv sync --locked`, so commit
`uv.lock` with every dependency change (`uv add <pkg>`, `uv add --dev <pkg>`).

## Models

Model paths, the manifest and provisioning are in
[`.claude/rules/models.md`](.claude/rules/models.md). The manifest
`src/transcribe/models/model_manifest.json` is the single source of truth.

## Changing the repo

- **Behaviour or feature:** update the tests, then `docs/design.md` (the
  contract, plus a dated decision-log entry if a decision changed),
  `docs/how-it-works.md` if a stage changed, and the README if users see it.
- **`Settings` default:** ask first. Tune on held-out audio, confirm on the
  evaluation recording (see `docs/benchmark.md`), add the dated table to
  `docs/benchmark.md`, a decision-log entry, the `Settings` docstring, and
  the README summary if headline numbers move.
- **Model:** ask first. Follow the
  [manifest procedure](.claude/rules/models.md), update the
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
- mypy runs in `strict` mode, so tests and scripts need full annotations too.
  Untyped engine packages (`faster_whisper`, `sherpa_onnx`, `mlx_whisper`) are
  exempted in `pyproject.toml`; keep their untyped surface inside the adapters.
- Python 3.13 evaluates annotations eagerly: modules that import types under
  `if TYPE_CHECKING:` need `from __future__ import annotations`.
- pytest runs with `filterwarnings = ["error"]` (any warning fails a test),
  `xfail_strict`, `--strict-markers` and `--import-mode=importlib`. Tests
  that need models carry the `engine` marker.
- Golden files under `tests/pipeline/steps/test_rendering/` come from
  `pytest-regressions`: regenerate them with `--force-regen` and never edit
  them by hand (they have their own markdownlint config).

## Markdown

Linting and citation style for docs are in
[`.claude/rules/markdown.md`](.claude/rules/markdown.md).

## Using tokens well

Each request re-sends the whole conversation, so keep it small:

- Read the part you need: list a long doc's headings
  (`grep -n '^## ' docs/benchmark.md`), then read that section. Prefer a search
  or a line range to reading a whole file.
- Run benchmarks and analyse `bench/` through the `bench-analyst` agent.
- Before long background work, state the expected finish time. Never schedule a
  wake-up or end a turn only to report progress.
- Suggest `/clear` when the task changes or a plan is approved (the plan file
  keeps the plan), and a short handoff note before a break of over an hour.
- Suggest the model at the start of a task, since switching later rebuilds the
  cache: Sonnet for docs, tests and small fixes; Opus for design and diarization
  work. Suggest `/effort low` for trivial changes.

## Compact instructions

When compacting, keep benchmark gate numbers, files changed, decisions awaiting
the user, and the checks still to run.

## Commits

Use the repo's `caveman-commit` skill (`.claude/skills/caveman-commit/`) for
terse Conventional Commits messages, one logical change per commit.
