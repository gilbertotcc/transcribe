"""Engine-independent records. Times are seconds on the original, untrimmed recording."""

import math
from dataclasses import dataclass


TIME_TOLERANCE = 0.001  # seconds an engine may report `end` before `start`


def clamp_interval(start: float, end: float, duration: float) -> tuple[float, float]:
    """Clip an engine interval to [0, duration] and repair `end < start` within the tolerance."""
    _check_finite(start, end)
    start = min(max(start, 0.0), duration)
    end = min(max(end, 0.0), duration)
    if start - TIME_TOLERANCE <= end < start:
        end = start
    return start, end


def _check_finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        msg = f"time must be finite, got {values}"
        raise ValueError(msg)


def _check_interval(start: float, end: float) -> None:
    _check_finite(start, end)
    if start < 0:
        msg = f"start must not be negative, got {start}"
        raise ValueError(msg)
    if end < start:
        msg = f"end ({end}) must not precede start ({start})"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Word:
    start: float
    end: float
    text: str  # verbatim from the engine, including its leading space

    def __post_init__(self) -> None:
        _check_interval(self.start, self.end)


@dataclass(frozen=True, slots=True)
class Turn:
    start: float
    end: float
    speaker: int  # opaque engine cluster ID

    def __post_init__(self) -> None:
        _check_interval(self.start, self.end)


@dataclass(frozen=True, slots=True)
class AttributedWord:
    word: Word
    speaker: int | None  # None: no diarization turn covers the word
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class Language:
    code: str  # a key of `languages.LANGUAGES`
    probability: float | None  # detection confidence; None when the user forced the language

    @property
    def forced(self) -> bool:
        return self.probability is None


@dataclass(frozen=True, slots=True)
class Paragraph:
    start: float
    speaker_label: str
    text: str

    def __post_init__(self) -> None:
        _check_interval(self.start, self.start)
