"""Assign words to diarization speakers and group them into paragraphs (pure)."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transcribe.domain import AttributedWord, Paragraph


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from transcribe.domain import Turn, Word

UNKNOWN_LABEL = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class _SpeakerTimeline:
    """One speaker's turns, merged into disjoint intervals sorted by time."""

    speaker: int
    first_start: float
    starts: list[float]
    ends: list[float]

    def overlap(self, start: float, end: float) -> float:
        total = 0.0
        index = bisect_left(self.starts, end) - 1  # last interval starting before `end`
        while index >= 0 and self.ends[index] > start:
            total += min(end, self.ends[index]) - max(start, self.starts[index])
            index -= 1
        return total

    def contains(self, time: float) -> bool:
        index = bisect_right(self.starts, time) - 1
        return index >= 0 and self.ends[index] >= time


def _timelines(turns: Iterable[Turn]) -> list[_SpeakerTimeline]:
    by_speaker: dict[int, list[Turn]] = {}
    for turn in turns:
        by_speaker.setdefault(turn.speaker, []).append(turn)
    timelines = []
    for speaker, speaker_turns in by_speaker.items():
        speaker_turns.sort(key=lambda turn: turn.start)
        starts: list[float] = []
        ends: list[float] = []
        for turn in speaker_turns:
            if ends and turn.start <= ends[-1]:
                ends[-1] = max(ends[-1], turn.end)
            else:
                starts.append(turn.start)
                ends.append(turn.end)
        timelines.append(_SpeakerTimeline(speaker, starts[0], starts, ends))
    # Tie-break order: earliest turn start, then speaker ID.
    timelines.sort(key=lambda timeline: (timeline.first_start, timeline.speaker))
    return timelines


def attribute(words: Iterable[Word], turns: Iterable[Turn]) -> list[AttributedWord]:
    """Give each word the speaker with the largest overlap; no overlap means unknown."""
    timelines = _timelines(turns)
    attributed = []
    for word in words:
        if word.end > word.start:
            scores = [(timeline.overlap(word.start, word.end), timeline.speaker) for timeline in timelines]
            best = max((score for score, _ in scores), default=0.0)
            candidates = [speaker for score, speaker in scores if score > 0 and score == best]
        else:
            candidates = [timeline.speaker for timeline in timelines if timeline.contains(word.start)]
        # `timelines` is already in tie-break order, so the first candidate wins.
        speaker = candidates[0] if candidates else None
        attributed.append(AttributedWord(word, speaker, ambiguous=len(candidates) > 1))
    return attributed


def speaker_label(number: int) -> str:
    return f"SPEAKER_{number:02d}"


def paragraphs(words: Sequence[AttributedWord], *, max_gap: float, max_duration: float) -> list[Paragraph]:
    """Group consecutive words by label; also break on long gaps and long paragraphs."""
    labels: dict[int, str] = {}
    result: list[Paragraph] = []
    current: list[AttributedWord] = []
    current_label = ""

    def flush() -> None:
        text = "".join(item.word.text for item in current).strip()
        if text:
            result.append(Paragraph(current[0].word.start, current_label, text))
        current.clear()

    for item in words:
        if item.speaker is None:
            label = UNKNOWN_LABEL
        else:
            label = labels.setdefault(item.speaker, speaker_label(len(labels) + 1))
        if current and (
            label != current_label
            or item.word.start - current[-1].word.end > max_gap
            or item.word.end - current[0].word.start > max_duration
        ):
            flush()
        current.append(item)
        current_label = label
    if current:
        flush()
    return result
