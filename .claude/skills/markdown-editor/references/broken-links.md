# Broken links

How to handle a link lychee reports as unreachable.

1. **Recheck it.** Open the URL with a web fetch tool, if the agent has one,
   or `curl -sIL <url>`, to tell a dead link from a flaky host.
2. **Replace a dead link.** Search the web for the same content at a working
   URL (a moved page, an archived copy, a pinned tag) and swap it in.
3. **Exclude a link that works but fails the check** (auth wall, rate limit,
   bot blocking) only after the user confirms. Add it to the project's
   lychee ignore mechanism, usually `.lycheeignore` or `exclude` in
   `lychee.toml` (see
   [excluding links](https://lychee.cli.rs/recipes/excluding-links/)), with
   a dated comment saying who verified access, in one of these forms:

   ```text
   # Access to the URL was manually verified by <git user>. (Last access: YYYY-MM-DD)
   ```

   ```text
   # Access to the URL was verified by <agent + model> using <tool>. (Last access: YYYY-MM-DD)
   ```

   Use the first form when the user checked the URL (e.g. in a browser),
   the second when the agent checked it with a tool. Fill in the real git
   user (`git config user.name`), agent, model, tool and today's date; the
   committed comment contains no angle-bracket placeholders.
4. **No ignore mechanism?** Tell the user the link stays broken and suggest
   adding the starter `.lycheeignore` (see [setup.md](setup.md)).
