from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from transcribe.domain import Language, Paragraph
from transcribe.rendering import escape, render_markdown, timestamp


if TYPE_CHECKING:
    from pytest_regressions.file_regression import FileRegressionFixture


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "[00:00:00]"),
        (3.999, "[00:00:03]"),
        (59.5, "[00:00:59]"),
        (3600.0, "[01:00:00]"),
        (86_400 + 61.2, "[24:01:01]"),
        (100 * 3600 + 5, "[100:00:05]"),
    ],
)
def test_timestamp_floors_and_allows_large_hours(seconds: float, expected: str) -> None:
    assert timestamp(seconds) == expected


def test_escape_markdown_and_html_characters() -> None:
    raw = r"a\b `c` *d* _e_ [f](g) <h> i|j & #k"

    assert escape(raw) == r"a\\b \`c\` \*d\* \_e\_ \[f\](g) \<h\> i\|j \& \#k"


def test_escape_flattens_line_breaks() -> None:
    assert escape("one\ntwo\r\nthree\u2028four") == "one two three four"


def test_render_escapes_title_and_paragraphs(file_regression: FileRegressionFixture) -> None:
    document = render_markdown(
        "meeting *final* [v2].m4a",
        [
            Paragraph(3.7, "SPEAKER_01", "Welcome to the meeting."),
            Paragraph(8.0, "SPEAKER_02", "Let us begin: <b>item #1</b> & 50% of_the *budget*."),
            Paragraph(4000.2, "UNKNOWN", "Unattributed words."),
        ],
    )

    file_regression.check(document, extension=".md", newline="\n")


def test_render_no_speech(file_regression: FileRegressionFixture) -> None:
    file_regression.check(render_markdown("silence.wav", []), extension=".md", newline="\n")


@pytest.mark.parametrize("language", [Language("it", 0.97), Language("it", None)], ids=["detected", "forced"])
def test_render_language_line(file_regression: FileRegressionFixture, language: Language) -> None:
    document = render_markdown("riunione.m4a", [Paragraph(1.0, "SPEAKER_01", "Buongiorno a tutti.")], language)

    file_regression.check(document, extension=".md", newline="\n")
