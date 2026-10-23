# AGENTS.md

This file provides guidance to AI coding agents when working with code in this
repository.

## Project

`transcribe` is a CLI that extracts text from audio files using **local** AI
models only: no data may leave the machine. It should also run on older
hardware that modern apps often drop, so prefer dependencies and models with
modest CPU/RAM requirements and no hard GPU requirement.

The project is at the scaffolding stage: `src/transcribe/__init__.py` holds a
placeholder `main()` (the `transcribe` console script and `python -m
transcribe` both call it), and `tests/test_smoke.py` is the only test.

## Current work: implement the MVP

`idea/` holds the reviewed specifications to implement. Start any
implementation task by reading
[`idea/transcribe-mvp-agent-plan.md`](idea/transcribe-mvp-agent-plan.md)
(section 0 explains how to use it), then follow its milestones in order.

- The plan wins over this file where they differ on project facts, for
  example it moves the project from Python 3.14 to 3.13 (milestone M0). Update
  this file as each milestone changes those facts.
- Ask the user instead of changing a decision or stop gate in the plan.
- Treat files in `idea/` as specifications: edit them only to record agreed
  decisions or measured results.

## Commands

The project uses `uv` (>= 0.13.5) and Python 3.14 (`src/` layout, `uv_build`
backend).

```sh
uv sync                          # install project + dev dependencies
uv run transcribe                # run the CLI
uv run ruff check                # lint
uv run ruff format               # format (CI runs `ruff format --check`)
uv run mypy                      # type check (strict, covers src/ and tests/)
uv run pytest                    # run all tests
uv run pytest tests/test_smoke.py::test_main_greets   # run a single test
```

CI (`.github/workflows/check-python.yml`) runs `uv sync --locked`, so commit
`uv.lock` whenever dependencies change (`uv add <pkg>` /
`uv add --dev <pkg>`).

## Conventions enforced by tooling

- Ruff has a broad rule set (see `pyproject.toml`), with 120-char lines. Notable
  rules: `T20` forbids `print` (silence with a justified `# noqa: T201`),
  `PTH` requires `pathlib`, `FBT` forbids boolean positional args, `TC` moves
  type-only imports under `if TYPE_CHECKING:`, and isort wants two blank lines
  after imports.
- mypy runs in `strict` mode, so tests need full type annotations too.
- pytest runs with `filterwarnings = ["error"]` (any warning fails a test),
  `xfail_strict`, and `--import-mode=importlib`. `pytest-regressions` is
  available for snapshot-style tests.

## Markdown

All `*.md` files (except under `.claude/`) are linted in CI by
`markdownlint-cli2` (`.markdownlint-cli2.yaml`: 80-char lines outside tables
and code blocks, ATX headings, aligned tables) and link-checked by `lychee`.

## Commits

The repo ships a `caveman-commit` skill (`.claude/skills/caveman-commit/`) for
terse Conventional Commits messages; use it when writing commit messages.
