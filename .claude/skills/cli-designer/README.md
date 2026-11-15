# cli-designer

An agent skill to design, validate and extend command-line interfaces with
the [Command Line Interface Guidelines](https://clig.dev/) (clig.dev). It
defines new commands and subcommands, reviews a CLI or a diff against the
guidelines, and suggests missing commands and flags. A project's own CLI
guidelines take precedence over clig.dev. The skill also lists recommended
CLI libraries for common languages.

## Install

```sh
npx skills add gilbertotcc/agent-skills --skill cli-designer
```

## Requirements

None. The agent reads the live guidelines from `https://clig.dev/llms.txt`
when it has network access, and otherwise uses the skill's bundled checklist.

## Data use

- The agent fetches `https://clig.dev/llms.txt`, if network access is
  available.
- It runs your CLI only with harmless calls (`--help`, `--version`).
- Apart from the clig.dev fetch, the skill makes no network requests of its
  own. Your CLI may still send data by itself, even for `--help` or
  `--version`.
