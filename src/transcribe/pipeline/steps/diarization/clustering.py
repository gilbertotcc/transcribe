"""Re-cluster diarization turns from one speaker embedding per turn (pure; numpy only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from transcribe.domain import Turn


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    Vectors = npt.NDArray[np.float64]


def normalize(vectors: Vectors) -> Vectors:
    """Unit rows; all-zero rows (turns without an embedding) stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def _centroid(vectors: Vectors, weights: Vectors) -> Vectors:
    centroid: Vectors = normalize((weights[:, None] * vectors).sum(axis=0, keepdims=True))[0]
    return centroid


def average_linkage(vectors: Vectors, weights: Vectors, *, threshold: float, k: int | None = None) -> list[int]:
    """Group unit vectors by weighted average-linkage cosine distance.

    Merging stops when the closest groups are farther apart than `threshold`,
    or, when `k` is given, at `k` groups regardless of distance.
    Returns a group index per row, numbered by first row.
    """
    n = len(vectors)
    if n == 0:
        return []
    distance = 1.0 - vectors @ vectors.T
    np.fill_diagonal(distance, np.inf)
    weight = weights.astype(np.float64).copy()
    group = np.arange(n)
    groups = n
    while groups > 1 and (k is None or groups > k):
        i, j = divmod(int(np.argmin(distance)), n)
        if k is None and distance[i, j] > threshold:
            break
        merged = (weight[i] * distance[i] + weight[j] * distance[j]) / (weight[i] + weight[j])
        distance[i, :] = merged
        distance[:, i] = merged
        distance[i, i] = np.inf
        distance[j, :] = np.inf
        distance[:, j] = np.inf
        weight[i] += weight[j]
        group[group == j] = i
        groups -= 1
    numbering: dict[int, int] = {}
    return [numbering.setdefault(int(g), len(numbering)) for g in group]


def recluster(
    turns: Sequence[Turn],
    embeddings: Vectors,
    *,
    threshold: float,
    anchor_seconds: float,
    join_similarity: float,
    fold_seconds: float,
    fold_similarity: float,
    k: int | None = None,
) -> list[Turn]:
    """Relabel turns; `embeddings` has one unit row per turn, all zeros where a turn has none.

    1. Turns of at least `anchor_seconds` with an embedding are anchors; average
       linkage groups them up to `threshold` (or into `k` groups), ignoring the
       engine's labels.
    2. The other turns, taken together per engine label, join the most similar
       group if their cosine similarity reaches `join_similarity`; otherwise they
       form a group of their own.
    3. Groups with less than `fold_seconds` of speech fold into their most similar
       larger group when the similarity reaches `fold_similarity` (0 s: no folding).
    """
    if not turns:
        return []
    durations = np.array([turn.end - turn.start for turn in turns])
    has_vector = np.linalg.norm(embeddings, axis=1) > 0
    anchor = has_vector & (durations >= anchor_seconds)
    # Units: each anchor turn alone; the remaining turns of one engine label together.
    unit_of_turn = [index if anchor[index] else -1 - turn.speaker for index, turn in enumerate(turns)]
    members: dict[int, list[int]] = {}
    for index, unit in enumerate(unit_of_turn):
        members.setdefault(unit, []).append(index)
    anchored = sorted(unit for unit in members if unit >= 0)

    anchor_vectors = embeddings[anchored]
    anchor_weights = durations[anchored]
    group_of_anchor = average_linkage(anchor_vectors, anchor_weights, threshold=threshold, k=k)
    group_of_unit = dict(zip(anchored, group_of_anchor, strict=True))
    n_groups = max(group_of_anchor, default=-1) + 1
    labels_array = np.array(group_of_anchor)
    centroids = [
        _centroid(anchor_vectors[labels_array == group], anchor_weights[labels_array == group])
        for group in range(n_groups)
    ]
    for unit in sorted(unit for unit in members if unit < 0):
        rows = [i for i in members[unit] if has_vector[i]]
        if rows and centroids:
            similarities = np.array(centroids) @ _centroid(embeddings[rows], durations[rows])
            if float(np.max(similarities)) >= join_similarity:
                group_of_unit[unit] = int(np.argmax(similarities))
                continue
        group_of_unit[unit] = n_groups
        n_groups += 1
    labels = [group_of_unit[unit] for unit in unit_of_turn]
    if fold_seconds > 0:
        labels = fold_small_groups(labels, embeddings, durations, fold_seconds, fold_similarity)
    return [Turn(turn.start, turn.end, label) for turn, label in zip(turns, labels, strict=True)]


def fold_small_groups(
    labels: Sequence[int], embeddings: Vectors, durations: Vectors, min_seconds: float, similarity: float
) -> list[int]:
    """Relabel groups with less than `min_seconds` of speech as their most similar larger group, if similar enough."""
    groups = sorted(set(labels))
    rows = {g: [i for i, label in enumerate(labels) if label == g] for g in groups}
    with_vector = {g: [i for i in rows[g] if np.linalg.norm(embeddings[i]) > 0] for g in groups}
    large = [g for g in groups if durations[rows[g]].sum() >= min_seconds and with_vector[g]]
    if not large:
        return list(labels)
    centroids = np.array([_centroid(embeddings[with_vector[g]], durations[with_vector[g]]) for g in large])
    target: dict[int, int] = {}
    for g in groups:
        if g in large or not with_vector[g]:
            continue
        similarities = centroids @ _centroid(embeddings[with_vector[g]], durations[with_vector[g]])
        if float(np.max(similarities)) >= similarity:
            target[g] = large[int(np.argmax(similarities))]
    return [target.get(label, label) for label in labels]
