---
paths:
  - "**/*.md"
---

# Markdown

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
