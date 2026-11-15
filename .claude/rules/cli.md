---
paths:
  - "src/transcribe/cli/**"
  - "tests/cli/**"
  - "docs/design.md"
---

# CLI

Design, review and extend the CLI with the `cli-designer` skill. Its project
guidelines are the [CLI contract](../../docs/design.md#cli-contract) and the
hard rules in [`AGENTS.md`](../../AGENTS.md); they win over clig.dev.

These deviations from clig.dev are decided. Treat them as `[project]` rules:
don't propose them and don't report them as findings.

- Exit codes are grouped (0, 1, 2, 130): a missing model and a bad path share
  code 2 with usage errors.
- stdout carries only the Markdown document: no `--json` or `--plain`.
- Output is never overwritten: no `-f`/`--force`, and `-` is not stdin or
  stdout.
- The frozen `Settings` dataclass is the only seam for tunables: no
  environment variables or configuration files.
- No debug file or `--debug` flag: error messages name the stage and the
  exception type, never recognized text.

A proposal from the skill is still a change to the CLI surface: ask the user
first.
