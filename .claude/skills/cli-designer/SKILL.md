---
name: cli-designer
description: >-
  Design, validate and extend command-line interfaces with the Command Line
  Interface Guidelines (clig.dev). Use to define a command or subcommand,
  review a CLI, or suggest missing commands.
license: MIT
compatibility: >-
  Network access is optional: the skill reads clig.dev live and falls back
  to its bundled checklist.
---

# CLI designer

Design, review and extend a command-line interface following the
[Command Line Interface Guidelines](https://clig.dev/) (clig.dev). There are
three modes: **design** a command, **validate** a CLI, **suggest** commands.

## Guideline sources

When sources disagree, the higher one wins:

1. **Project guidelines.** Find the CLI rules the project already writes
   down. Check its agent instructions, contributing guide and design or
   style docs, search its documentation for terms like "CLI", "command
   line" or "flags", and follow the links they give. A project rule replaces
   the clig.dev rule it contradicts. Apply it in the project's own form: if
   the project uses `--format json`, never propose `--json`.
2. **clig.dev, read live.** Before you deliver, read the live sections behind
   your decisions or findings, once per task. Don't read the whole file:
   `llms.txt` is about 72 KB. The section headings are `The Basics`, `Help`,
   `Documentation`, `Output`, `Errors`, `Arguments and flags`,
   `Interactivity`, `Subcommands`, `Robustness`, `Future-proofing`,
   `Signals and control characters`, `Configuration`,
   `Environment variables`, `Naming`, `Distribution` and `Analytics`:

   ```sh
   curl -sL https://clig.dev/llms.txt | awk '/^## /{g=($0 ~ /^## Guidelines/)}
     g && /^### (Output|Errors)( \{|$)/{p=1;print;next} /^##+ /{p=0} p'
   ```

   Replace `Output|Errors` with the sections you need.
3. **Bundled checklist.** [references/checklist.md](references/checklist.md)
   is the list to walk through, and the fallback when clig.dev can't be read.
   If clig.dev can't be read, tell the user.

Tag each rationale item and each finding with its source: `[project]`, or
`[clig: <section>]` if you read it live, else `[checklist: <section>]`.
Never rely on clig.dev from memory.

## Before any mode

- Map the command tree. Run the bare CLI, then `<cli> --help` and
  `<cli> <cmd> --help` for each subcommand, or read the command definitions
  in the code. Run only harmless calls. A crash is a finding, not a blocker.
- Identify the parsing library from the manifest (`pyproject.toml`,
  `go.mod`, `Cargo.toml`, `package.json`, `pom.xml`, `*.csproj`, `Gemfile`).
  Note the CLI's existing conventions: command order, flag names, output
  format flag and exit codes. They win over the checklist's defaults.
- If the project has no CLI yet, propose a library from
  [references/libraries.md](references/libraries.md).

## Design a command

1. Work out what the operation needs: inputs, outputs, side effects, data it
   relies on, and how bad it is if it goes wrong. Severity, from the
   checklist's Arguments and flags section: mild, moderate or severe.
2. Decide the interface, using the checklist sections Arguments and flags,
   Subcommands, Output and Interactivity:
   - the name and its place in the tree, keeping the existing order and
     verb set;
   - flags over positional arguments, standard or existing flag names, and a
     long form for every flag;
   - the stdout/stderr split, the machine-readable output flag, and exit
     codes (reuse existing ones; otherwise 0 ok, 1 error, 2 usage, and a
     distinct code for each specific failure, such as not found);
   - confirmation scaled to severity, plus `--force`, `--dry-run` and
     `--no-input` where they apply;
   - how secrets get in (a file or stdin, never a flag or env var).
3. Deliver:
   - a synopsis;
   - draft `--help` text that starts with examples;
   - behavior: streams, prompts, JSON shape, exit codes;
   - companion changes: existing commands affected, and the undo or
     inverse command;
   - open questions, e.g. missing data fields;
   - a short rationale, with tags.

   Implement it only when the user asks, in the project's library.

## Validate a CLI

1. Scope the review: the diff, the commands the user names, or the whole
   CLI when asked. Check the change itself, not the codebase in general.
2. Walk each checklist section against it. Observe real behavior with
   harmless calls only: `--help`, `--version`, no arguments, a typo, piping
   to `cat` (color and animation off), and checking which stream errors go
   to. Never run commands that change state. If a call is blocked or
   unsafe, read the code instead.
3. Report findings ordered by severity across all sections:
   - **High**: security, data loss, or broken scripting (secrets in flags,
     errors that exit `0`, prompts with no TTY);
   - **Medium**: wrong streams, missing machine output, inconsistent names
     or flags, crashes on bad input;
   - **Low**: help text, polish.

   Each finding gives `file:line` (or the command run), the problem, the
   tagged rule, and a fix. Merge findings that share a root cause or a fix.
   End with one line listing N/A sections and why. Don't list passes.

## Suggest commands

1. Inventory the commands, flags and env vars the CLI has.
2. Compare them with the checklist's **(surface)** items, translated into
   the project's conventions, and with consistency gaps: a noun missing a
   verb its siblings have (`list`/`get`/`create`/`delete`), a flag named
   differently across commands, or no machine-readable output.
3. Deliver at most about 10 items, ordered security, then correctness, then
   consistency, then convenience. Each item gives a synopsis, the reason
   with its tag, and its kind: flag, new command, behavior fix, or breaking
   change. For breaking changes, give a deprecation path (see
   Future-proofing). Mention defects you notice in one line each, and offer
   a validation.

## Boundaries

- Report and propose first. Edit code only when the user asks after seeing
  the findings, then re-check only what changed.
- Keep proposals in the project's language, library and style. Never
  switch the parsing library unless asked.

## Example

Request: "Add a way to delete a bookmark" to a CLI with
`bookmarks list|get [--json]`.

````text
Synopsis: mycli bookmarks delete <id>... [--force] [--dry-run] [--json]
Examples:
  mycli bookmarks delete 42
  mycli bookmarks list --json | jq -r '.[].id' | mycli bookmarks delete -

Options:
  -f, --force     Don't ask for confirmation (required without a TTY)
  -n, --dry-run   Show what would be deleted
      --json      Print deleted IDs as JSON (same flag as list/get)

Behavior: y/N prompt on stderr; IDs on stdout; exit 0/1/2, 3 not found.
Rationale: noun-verb like its siblings [clig: Subcommands]; moderate
deletion asks y/N, --force for scripts [clig: Arguments and flags].
````
