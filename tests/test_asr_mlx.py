"""Pure parts of the mlx-whisper adapter (no MLX, no models)."""

from __future__ import annotations

from transcribe.asr_mlx import group_chunks


def test_chunks_fill_groups_up_to_the_limit() -> None:
    chunks = [{"start": 0, "end": 40}, {"start": 50, "end": 90}, {"start": 100, "end": 130}]

    assert group_chunks(chunks, 80) == [chunks[:2], chunks[2:]]


def test_every_chunk_lands_in_exactly_one_group() -> None:
    chunks = [{"start": 100 * i, "end": 100 * i + 60} for i in range(7)]

    groups = group_chunks(chunks, 150)

    assert [chunk for group in groups for chunk in group] == chunks
    assert all(sum(c["end"] - c["start"] for c in group) <= 150 for group in groups)


def test_a_chunk_longer_than_the_limit_is_a_group_of_its_own() -> None:
    chunks = [{"start": 0, "end": 10}, {"start": 20, "end": 500}, {"start": 510, "end": 520}]

    assert group_chunks(chunks, 100) == [[chunks[0]], [chunks[1]], [chunks[2]]]


def test_no_speech_gives_no_groups() -> None:
    assert group_chunks([], 100) == []
