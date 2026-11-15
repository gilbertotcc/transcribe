---
name: markdown-editor
description: >-
  Write and edit a project's Markdown files in its own markdownlint-cli2
  style and check their links. Use for project .md files, not for issue/PR
  bodies or other text bound for external tools.
license: MIT
compatibility: >-
  Needs markdownlint-cli2, or Node.js with npx (first run downloads it).
  lychee (link checks) is optional.
---

# Markdown editor

## Scope and style

Covers the project's own Markdown files. Run every command from the project
root (the git top level, else the directory holding the config): the tools
read the root config and its ignores only from there. Style is whatever
that config says, else markdownlint's defaults; never apply a remembered
style, and let no other formatter rewrite Markdown.

Text bound for an external tool (issue, PR or comment body, chat message,
notes for the user) stays out:

- draft it outside the project and pass it directly, e.g.
  `gh issue create --body-file <tmp>`;
- write one line per paragraph or list item;
- don't lint or link-check it.

## Workflow

After each change to a project `.md` file, before calling the task done:

```sh
markdownlint-cli2 --no-globs --fix <file>  # 1. lint
lychee --no-progress <file>                # 2. links
```

1. **Lint.** `--no-globs` limits the run to the file. The tool reads
   `<file>` as a glob: backslash-escape `* ? [ ] ( )` in it (e.g.
   `docs/\[draft\].md`), and report a name with braces unlinted. Exit 1 means issues
   remain: each names a rule and line, which the
   [markdownlint rules](https://github.com/DavidAnson/markdownlint/blob/v0.42.1/doc/Rules.md)
   explain. Edit the content (don't disable the rule) and rerun until it
   exits 0. "Linting: 0 files" means the config ignores the file.
2. **Check links** once the content is final. Fix or remove broken ones;
   for links that look fine but fail, follow
   [references/broken-links.md](references/broken-links.md).
3. **Repeat** both steps if anything changed.

If a tool is missing, tell the user what wasn't checked and point them to
the skill's `README.md`. Install it, or run `npx --yes markdownlint-cli2`
(which downloads it), only when the user explicitly asks and no project or
agent configuration (sandbox, permissions) blocks it.

## Sweep the project

```sh
markdownlint-cli2 --fix                            # config has globs
markdownlint-cli2 --fix "**/*.md" "#node_modules"  # it doesn't
lychee --no-progress --hidden --extensions md .
```

`--hidden` includes dot-directories such as `.github`; the config's
`exclude_path` still applies.

Drop `--fix` to only report. If the config sets `fix: true`, every run
fixes in place, so review the diff. Report files checked, issues fixed and
issues left.

## Setup

Only when the user asks, or agrees after finding no config: follow
[references/setup.md](references/setup.md).

## Writing content

- Give every document exactly one top-level (`#`) heading.
- Tag fenced code blocks with their language unless the config disables
  MD040.
- Use relative links for files in the same repository.
- Keep each emphasized span (`*...*`, `**...**`) on one line when wrapping.
- Cite sources with the project's own convention (`AGENTS.md`, contributing
  docs, existing files), else
  [references/citations.md](references/citations.md).
- Add a spelling or prose-style pass only when the project asks for one.
