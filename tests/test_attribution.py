from __future__ import annotations

import pytest

from transcribe.attribution import attribute, paragraphs, speaker_label
from transcribe.domain import AttributedWord, Paragraph, Turn, Word


def speakers(words: list[Word], turns: list[Turn]) -> list[tuple[int | None, bool]]:
    return [(item.speaker, item.ambiguous) for item in attribute(words, turns)]


def test_largest_overlap_wins() -> None:
    turns = [Turn(0.0, 1.2, 1), Turn(1.2, 3.0, 2)]

    assert speakers([Word(1.0, 2.0, " x")], turns) == [(2, False)]


def test_no_turns_gives_unknown() -> None:
    assert speakers([Word(0.0, 1.0, " x")], []) == [(None, False)]


def test_no_overlap_gives_unknown_without_carry_over() -> None:
    turns = [Turn(0.0, 1.0, 5)]
    words = [Word(0.2, 0.8, " a"), Word(1.0, 1.5, " b")]  # touching the end is not overlap

    assert speakers(words, turns) == [(5, False), (None, False)]


def test_same_speaker_overlapping_turns_are_not_double_counted() -> None:
    # Speaker 1 has two overlapping turns covering 0.4 s of the word; speaker 2 covers 0.5 s.
    turns = [Turn(0.0, 0.4, 1), Turn(0.1, 0.4, 1), Turn(0.5, 1.0, 2)]

    assert speakers([Word(0.0, 1.0, " x")], turns) == [(2, False)]


def test_tie_prefers_earliest_turn_start_and_is_ambiguous() -> None:
    turns = [Turn(5.0, 6.0, 0), Turn(0.0, 0.5, 9), Turn(5.5, 6.5, 9)]  # 9 starts earlier overall

    assert speakers([Word(5.5, 6.0, " x")], turns) == [(9, True)]


def test_tie_with_equal_first_start_prefers_lower_speaker_id() -> None:
    turns = [Turn(1.0, 2.0, 4), Turn(1.0, 2.0, 2)]

    assert speakers([Word(1.0, 2.0, " x")], turns) == [(2, True)]


def test_zero_duration_word_uses_containing_turn() -> None:
    turns = [Turn(0.0, 1.0, 3), Turn(2.0, 3.0, 4)]
    words = [Word(2.0, 2.0, " a"), Word(1.5, 1.5, " b"), Word(1.0, 1.0, " c")]

    assert speakers(words, turns) == [(4, False), (None, False), (3, False)]


def test_zero_duration_word_in_overlap_is_ambiguous() -> None:
    turns = [Turn(0.0, 2.0, 1), Turn(1.0, 3.0, 0)]

    assert speakers([Word(1.5, 1.5, " x")], turns) == [(1, True)]


def test_many_turns_scale_linearly_enough() -> None:
    turns = [Turn(i * 1.0, i * 1.0 + 0.9, i % 3) for i in range(20_000)]
    words = [Word(i * 0.5, i * 0.5 + 0.3, " w") for i in range(40_000)]

    result = attribute(words, turns)

    assert len(result) == 40_000
    assert result[2].speaker == 1


def word(start: float, end: float, text: str, speaker: int | None) -> AttributedWord:
    return AttributedWord(Word(start, end, text), speaker, ambiguous=False)


def group(words: list[AttributedWord]) -> list[Paragraph]:
    return paragraphs(words, max_gap=2.0, max_duration=30.0)


def test_paragraphs_concatenate_words_verbatim_and_strip() -> None:
    words = [word(0.0, 0.5, " Hello", 0), word(0.5, 1.0, ",", 0), word(1.0, 1.5, " world. ", 0)]

    assert group(words) == [Paragraph(0.0, "SPEAKER_01", "Hello, world.")]


def test_labels_follow_first_appearance() -> None:
    words = [word(0.0, 1.0, " a", 42), word(1.0, 2.0, " b", 7), word(2.0, 3.0, " c", 42), word(3.0, 4.0, " d", None)]

    assert [(p.speaker_label, p.text) for p in group(words)] == [
        ("SPEAKER_01", "a"),
        ("SPEAKER_02", "b"),
        ("SPEAKER_01", "c"),
        ("UNKNOWN", "d"),
    ]


@pytest.mark.parametrize(("number", "label"), [(1, "SPEAKER_01"), (10, "SPEAKER_10"), (101, "SPEAKER_101")])
def test_labels_have_two_digits_minimum(number: int, label: str) -> None:
    assert speaker_label(number) == label


@pytest.mark.parametrize(("gap", "split"), [(2.0, False), (2.01, True)])
def test_gap_over_threshold_splits(gap: float, split: bool) -> None:
    words = [word(0.0, 1.0, " a", 0), word(1.0 + gap, 2.0 + gap, " b", 0)]

    assert len(group(words)) == (2 if split else 1)


def test_paragraph_longer_than_30_seconds_splits() -> None:
    words = [word(0.0, 10.0, " a", 0), word(10.0, 20.0, " b", 0), word(20.0, 30.0, " c", 0), word(30.0, 40.0, " d", 0)]

    result = group(words)

    assert [(p.start, p.text) for p in result] == [(0.0, "a b c"), (30.0, "d")]


def test_whitespace_only_paragraph_is_dropped() -> None:
    assert group([word(0.0, 1.0, " ", 0)]) == []


def test_no_words_gives_no_paragraphs() -> None:
    assert group([]) == []
