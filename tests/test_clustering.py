"""Re-clustering of diarization turns from per-turn embeddings (pure, synthetic embeddings)."""

from __future__ import annotations

from typing import Any

import numpy as np

from transcribe.domain import Turn
from transcribe.pipeline.steps.diarization import clustering


DEFAULTS: dict[str, Any] = {
    "threshold": 0.3,
    "anchor_seconds": 1.0,
    "join_similarity": 0.5,
    "fold_seconds": 0.0,
    "fold_similarity": 0.5,
}


def voices(*means: list[float], noise: float = 0.0, per_voice: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    rows = [np.array(mean) + noise * rng.standard_normal(len(mean)) for mean in means for _ in range(per_voice)]
    vectors: np.ndarray = clustering.normalize(np.array(rows, dtype=np.float64))
    return vectors


def turns_of(speakers: list[int], seconds: float = 3.0) -> list[Turn]:
    return [Turn(index * 5.0, index * 5.0 + seconds, speaker) for index, speaker in enumerate(speakers)]


def labels(turns: list[Turn], vectors: np.ndarray, **overrides: Any) -> list[int]:
    return [turn.speaker for turn in clustering.recluster(turns, vectors, **(DEFAULTS | overrides))]


def test_normalize_keeps_missing_embeddings_zero() -> None:
    vectors = clustering.normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 0.0]])


def test_average_linkage_merges_close_vectors_only() -> None:
    vectors = voices([1, 0, 0], [0, 1, 0], noise=0.05)
    assert clustering.average_linkage(vectors, np.ones(6), threshold=0.3) == [0, 0, 0, 1, 1, 1]


def test_average_linkage_stops_at_the_oracle_count() -> None:
    vectors = voices([1, 0, 0], [0, 1, 0], [0, 0, 1], noise=0.05)
    assert len(set(clustering.average_linkage(vectors, np.ones(9), threshold=0.0, k=2))) == 2
    assert clustering.average_linkage(np.zeros((0, 3)), np.zeros(0), threshold=0.3) == []


def test_recluster_merges_a_split_speaker_and_keeps_close_voices_apart() -> None:
    # The engine split voice A into speakers 0 and 1; voice B (2) is close to A but distinct.
    a, b = [1.0, 0.0, 0.0], [0.8, 0.6, 0.0]  # cosine distance 0.2
    result = labels(turns_of([0, 0, 0, 1, 1, 1, 2, 2, 2]), voices(a, a, b, noise=0.02), threshold=0.1)
    assert result[:6] == [result[0]] * 6
    assert result[6:] == [result[6]] * 3
    assert result[0] != result[6]


def test_recluster_ignores_mixed_engine_labels() -> None:
    result = labels(turns_of([0, 1, 0, 1, 0, 1]), voices([1, 0, 0], [0, 1, 0], noise=0.02))
    assert result == [0, 0, 0, 1, 1, 1]


def test_recluster_keeps_a_dissimilar_speaker_heard_only_in_short_turns() -> None:
    turns = turns_of([0, 0, 0]) + [Turn(20.0 + index, 20.5 + index, 1) for index in range(3)]
    result = labels(turns, voices([1, 0, 0], [0, 1, 0], noise=0.02))
    assert result[3:] == [result[3]] * 3
    assert result[3] != result[0]


def test_recluster_joins_similar_short_turns_and_keeps_turns_without_embeddings() -> None:
    vectors = voices([1, 0, 0], noise=0.02, per_voice=4)
    vectors[3] = 0.0  # too short to embed
    result = labels([*turns_of([0, 0]), Turn(20.0, 20.5, 1), Turn(30.0, 30.2, 2)], vectors)
    assert result[:3] == [result[0]] * 3
    assert result[3] != result[0]


def test_recluster_handles_one_speaker_and_no_turns() -> None:
    assert set(labels(turns_of([4, 4, 4]), voices([1, 0], noise=0.02))) == {0}
    assert clustering.recluster([], np.zeros((0, 2)), **DEFAULTS) == []


def test_recluster_without_anchors_keeps_the_engine_speakers_apart() -> None:
    turns = [Turn(0.0, 0.5, 0), Turn(1.0, 1.5, 1)]
    assert len(set(labels(turns, voices([1, 0], [0, 1], per_voice=1)))) == 2


def test_fold_moves_small_similar_groups_into_large_ones_only() -> None:
    vectors = voices([1, 0, 0], [0.95, 0.31, 0], [0, 1, 0], per_voice=1)
    durations = np.array([30.0, 2.0, 2.0])
    # Group 1 is small and similar to group 0; group 2 is small and dissimilar.
    assert clustering.fold_small_groups([0, 1, 2], vectors, durations, 10.0, 0.5) == [0, 0, 2]
    assert clustering.fold_small_groups([0, 1, 2], vectors, durations, 1.0, 0.5) == [0, 1, 2]  # nothing is small


def test_recluster_folds_a_brief_similar_speaker() -> None:
    a, a_drift = [1.0, 0.0, 0.0], [0.7, 0.71, 0.0]  # cosine distance about 0.3
    turns = [*turns_of([0, 0, 0], seconds=4.0), *[Turn(20.0 + 5 * i, 23.0 + 5 * i, 1) for i in range(2)]]
    vectors = np.vstack([voices(a, noise=0.01), voices(a_drift, noise=0.01, per_voice=2)])
    assert len(set(labels(turns, vectors, threshold=0.2))) == 2
    assert len(set(labels(turns, vectors, threshold=0.2, fold_seconds=8.0, fold_similarity=0.6))) == 1
