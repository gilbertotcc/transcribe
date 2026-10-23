# transcribe

`transcribe` is a Python script that extracts text from audio files. It runs
locally using local AI models, so no data leaves the machine you launch it on.
It should work on *older* machines that are often not supported by modern
applications. It is vibe-coded, meaning I accept all the goods and sh*ts AI can
produce.

## Status

Not usable yet. The MVP to build is described in
[`idea/transcribe-mvp-agent-plan.md`](idea/transcribe-mvp-agent-plan.md):
`transcribe AUDIOFILE [--output OUTPUT.md]` produces a Markdown transcript of
an English recording with anonymous speaker labels, on a CPU-only machine.
AI coding agents should start from [`AGENTS.md`](AGENTS.md).
