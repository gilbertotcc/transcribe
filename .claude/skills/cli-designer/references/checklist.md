# CLI checklist

A condensed version of the
[Command Line Interface Guidelines](https://clig.dev/), in the guidelines'
own section order. Each heading links to the full section. Read the live
guidelines for the reasoning behind an item, and let a project's own CLI
guidelines override any item here.

Items marked **(surface)** are commands, flags or variables a CLI is expected
to have. Use them when suggesting missing commands, renamed to the project's
conventions (e.g. `--format json` instead of `--json`).

## [The Basics](https://clig.dev/#the-basics)

- Use an argument-parsing library (see [libraries.md](libraries.md)).
- Exit `0` on success and non-zero on failure. Use distinct codes for distinct
  failures.
- Send primary output to stdout. Send messages, logs and errors to stderr.

## [Help](https://clig.dev/#help)

- **(surface)** `-h` and `--help` show full help on every command and
  subcommand, and mean nothing else.
- With no arguments, a command that needs some shows concise help: a
  description, one or two examples, and a pointer to `--help`.
- Help starts with examples and the most common flags and commands. It links
  to web docs and to a place to report issues.
- When the user makes a likely typo, suggest the right command. Don't run it
  without asking.
- A command that expects piped input shows help and exits if stdin is a TTY.

## [Documentation](https://clig.dev/#documentation)

- Provide web docs and terminal docs (`help <command>`, man pages if
  practical).

## [Output](https://clig.dev/#output)

- Human-readable output by default.
- **(surface)** `--json` for machine-readable output. Add `--plain` when the
  human format breaks line-based tools (`grep`, `awk`).
- On success, print brief output. `-q`/`--quiet` silences it.
- When state changes, say what changed. Make the current state easy to see,
  and suggest the next command when it helps.
- Ask before crossing boundaries: no network or file writes beyond what the
  user asked for.
- Turn color off when the stream (stdout or stderr) isn't a TTY, when
  `NO_COLOR` is non-empty, when `TERM=dumb`, or with **(surface)**
  `--no-color`. A non-empty `FORCE_COLOR` overrides the TTY and `TERM`
  checks, but `--no-color` and `NO_COLOR` still win.
- No animations or spinners when stdout isn't a TTY.
- No internal-only details by default, and don't log to stderr at debug
  level by default.
- Use a pager (`less -FIRX`) for long output, but only when stdout is a TTY.

## [Errors](https://clig.dev/#errors)

- Catch expected errors and rewrite them for humans, with a hint on how to
  fix the problem.
- Keep the signal-to-noise ratio high. Put the important information last,
  where the eye lands.
- For unexpected errors, write debug details to a file or show them behind a
  flag, and explain how to report a bug.

## [Arguments and flags](https://clig.dev/#arguments-and-flags)

- Prefer flags to positional arguments. Several arguments are fine only for
  the same kind of thing (e.g. `rm a b c`).
- Every flag has a long form. Short flags only for common options.
- Use standard flag names: `-a/--all`, `-d/--debug`, `-f/--force`, `--json`,
  `-h/--help`, `-n/--dry-run`, `--no-input`, `-o/--output`, `-p/--port`,
  `-q/--quiet`, `-u/--user`, `--version`. Avoid the ambiguous `-v`.
- The default is the right choice for most users.
- Prompt for missing input, but never *require* a prompt: every prompt has a
  flag equivalent.
- Confirm dangerous actions, scaled to severity:
  - *mild*: a small local change, such as deleting one file. A prompt is
    optional.
  - *moderate*: a bigger local change, a remote change, or a bulk change
    that is hard to undo. Ask `y/N`.
  - *severe*: deleting something complex, such as a whole app or server.
    Make the user type its name.

  **(surface)** `-f`/`--force`
  skips the confirmation in scripts. **(surface)** `-n`/`--dry-run` for
  risky changes.
- Accept `-` as a file argument to mean stdin or stdout.
- Optional values take a word such as `none`.
- Arguments, flags and subcommands work in any order where possible.
- Never take secrets via flags (they show up in `ps` and shell history).
  Use `--*-file` or stdin.

## [Interactivity](https://clig.dev/#interactivity)

- Prompt only when stdin is a TTY. **(surface)** `--no-input` turns all
  prompts off and fails with a hint about the flag to use instead.
- Don't echo passwords as the user types them.
- Ctrl-C always works. Say how to exit wrappers such as REPLs.

## [Subcommands](https://clig.dev/#subcommands)

- Same flag names, output formats and behavior across subcommands.
- Consistent levels: `noun verb` (more common) or `verb noun`, with the same
  verbs for every noun (`list`, `get`, `create`, `update`, `delete`).
- No ambiguous or similar names (e.g. `update` vs `upgrade`).

## [Robustness](https://clig.dev/#robustness-guidelines)

- Validate input early and fail before changing state.
- Print something within about 100 ms. Show progress for long operations
  (only on a TTY), and explain errors in parallel output clearly.
- Network calls have timeouts.
- Make operations recoverable (re-running resumes) and crash-only (no
  required cleanup). Make them idempotent where practical.
- Expect misuse: scripts, bad connections, many instances at once.

## [Future-proofing](https://clig.dev/#future-proofing)

- Keep changes additive. Warn and offer a migration path before you break
  anything.
- Machine-readable output, flags and subcommands are interfaces, so keep
  them stable. Human output can change.
- No catch-all default subcommand, and no arbitrary abbreviations. Aliases
  must be explicit and stable.
- No time bombs: don't depend on external services that may disappear.

## [Signals and control characters](https://clig.dev/#signals)

- On Ctrl-C, say so right away and exit fast, with a timeout on cleanup. A
  second Ctrl-C skips cleanup.
- Leave state that a later run can recover from after an interrupted
  cleanup.

## [Configuration](https://clig.dev/#configuration)

- Use flags for settings that vary per run, env vars for settings that vary
  with context, and versioned config files for stable project settings.
- Follow XDG (`~/.config/<app>`, `~/.cache/<app>`).
- Ask before changing configuration that belongs to another program, and say
  exactly what changes.
- Precedence: flags > env vars > project config (e.g. `.env`) > user config
  > system config. Document it.

## [Environment variables](https://clig.dev/#environment-variables)

- Names are `UPPER_SNAKE_CASE`, prefixed with the app's name. Don't take over
  widely used names.
- Values fit on one line.
- **(surface)** Honor `NO_COLOR` and `FORCE_COLOR`, `DEBUG`, `EDITOR`,
  `PAGER`, `HTTP(S)_PROXY`, `ALL_PROXY` and `NO_PROXY`, `TMPDIR`, `HOME`,
  `TERM`, `LINES` and `COLUMNS` where relevant.
- `.env` can supplement configuration but doesn't replace a config file.
- Never take secrets from env vars. Use files, stdin, or a secret store.

## [Naming](https://clig.dev/#naming)

- The command name is a simple, memorable word: short, lowercase, dashes only
  if needed, easy to type, not generic.

## [Distribution](https://clig.dev/#distribution)

- Ship a single binary or a standard package where possible.
- Make it easy to uninstall, and document how.

## [Analytics](https://clig.dev/#analytics)

- No telemetry or crash reports without explicit opt-in consent. Say what is
  collected and why.
- Prefer alternatives: docs analytics, download counts, asking users.

## Common practice (not in clig.dev)

- **(surface)** `--version` prints the version and exits `0`.
- **(surface)** For larger CLIs: shell completion (`completion <shell>`), and
  `config` subcommands to view and change settings.
- Releases follow semantic versioning, so breaking interface changes bump the
  major version.
