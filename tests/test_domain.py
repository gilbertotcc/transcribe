from __future__ import annotations

import math

import pytest

from transcribe.domain import Paragraph, Turn, Word, clamp_interval


@pytest.mark.parametrize(
    ("start", "end"),
    [(math.nan, 1.0), (0.0, math.inf), (-math.inf, 1.0), (-0.5, 1.0), (2.0, 1.0), (1.0, 1.0 - 0.0005)],
)
def test_word_rejects_invalid_times(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="time must be finite|negative|must not precede"):
        Word(start, end, " x")


def test_turn_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        Turn(3.0, 2.0, 0)


def test_paragraph_rejects_nan_start() -> None:
    with pytest.raises(ValueError, match="finite"):
        Paragraph(math.nan, "SPEAKER_01", "x")


def test_records_are_frozen() -> None:
    word = Word(0.0, 1.0, " x")

    with pytest.raises(AttributeError):
        word.start = 2.0  # type: ignore[misc]


def test_zero_duration_word_is_valid() -> None:
    assert Word(1.0, 1.0, " x").end == 1.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ((1.0, 2.0), (1.0, 2.0)),
        ((-0.2, 0.5), (0.0, 0.5)),  # before the recording starts
        ((9.5, 10.4), (9.5, 10.0)),  # past the end of the audio
        ((11.0, 12.0), (10.0, 10.0)),
        ((2.0, 1.9995), (2.0, 2.0)),  # end precedes start within 1 ms tolerance
        ((2.0, 1.5), (2.0, 1.5)),  # beyond tolerance: left for the record to reject
    ],
)
def test_clamp_interval(raw: tuple[float, float], expected: tuple[float, float]) -> None:
    assert clamp_interval(*raw, duration=10.0) == expected


def test_clamp_interval_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        clamp_interval(math.nan, 1.0, duration=10.0)
