# Citations

Default rules for projects with no citation convention of their own. Match
the weight of the citation to the source, and keep an open-world tone:
state what things do, adding a negation only where it prevents a likely
misreading.

## Web pages, docs, repositories and software

Link the source inline, where the claim appears, with descriptive link
text. Point to a pinned version (a tag or commit) when one exists, so the
link keeps saying what it said. No publication year is needed.

```markdown
Style comes from the [markdownlint-cli2 configuration](https://github.com/DavidAnson/markdownlint-cli2/tree/v0.24.1#configuration).
```

## Publications

For articles, papers, books, standards and reports, use APA 7: cite inline
as `([Author, Year])`, a shortcut reference link. When the year is unknown,
omit it (`([Author])`) instead of writing "n.d."; tell same-author,
same-year sources apart with letters (`2026a`). Link the DOI where there is
one.

## References section

End the document with a `## References` section when it cites publications,
or when some sources are relevant and significant enough to deserve a place
where readers can skim them all. List, alphabetically:

- each cited publication as `[Author, Year] <APA 7 reference>`, followed by
  its link definition `[Author, Year]: <url>`;
- each significant web source as a pinned link with a few words on what it
  covers; minor web sources stay inline only.

Keep each emphasized title on one line when wrapping.

```markdown
Readers skim headings first ([Nielsen, 2006]).

## References

- [Nielsen, 2006] Nielsen, J. (2006).
  *F-shaped pattern for reading web content*. Nielsen Norman Group.
- [markdownlint rules](https://github.com/DavidAnson/markdownlint/blob/v0.42.1/doc/Rules.md):
  what each lint rule expects.

[Nielsen, 2006]: https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content-discovered/
```
