"""Markdown rendering (pure: same input, byte-identical output)."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from transcribe.languages import label


if TYPE_CHECKING:
    from collections.abc import Sequence

    from transcribe.domain import Language, Paragraph

NOTICE = "> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted."
NO_SPEECH = "No speech detected."

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>|&#])")
_LINE_BREAKS = re.compile("[\r\n\v\f\x85\u2028\u2029]+")


def escape(text: str) -> str:
    """Make text render as plain prose: no line breaks, Markdown/HTML characters escaped."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", _LINE_BREAKS.sub(" ", text))


def timestamp(seconds: float) -> str:
    total = math.floor(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def render_markdown(source_name: str, paragraphs: Sequence[Paragraph], language: Language | None = None) -> str:
    """Render the transcript; `source_name` must already be a basename. No language, no language line."""
    blocks = [f"# Transcript: {escape(source_name)}"]
    if language is not None:
        blocks.append(f"Language: {label(language.code)}, {'forced' if language.forced else 'detected'}")
    blocks.append(NOTICE)
    if paragraphs:
        blocks.extend(f"{timestamp(p.start)} {p.speaker_label}: {escape(p.text)}" for p in paragraphs)
    else:
        blocks.append(NO_SPEECH)
    return "\n\n".join(blocks) + "\n"
