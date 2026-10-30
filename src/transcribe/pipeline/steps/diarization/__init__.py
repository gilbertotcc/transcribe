"""Speaker diarization: sherpa-onnx speaker turns, re-clustered from per-turn embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transcribe.pipeline.steps.diarization import clustering, sherpa


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from transcribe.config import Settings
    from transcribe.domain import Turn
    from transcribe.pipeline.events import ProgressCallback

_SHERPA_SHARE = 0.8  # of the stage's progress, when re-clustering follows (it takes about a fifth of the time)


def diarize(
    audio: npt.NDArray[np.float32], settings: Settings, *, progress: ProgressCallback | None = None
) -> list[Turn]:
    model = settings.recluster_embedding_model_path
    share = 1.0 if model is None else _SHERPA_SHARE

    def sherpa_progress(done: float, total: float) -> None:
        if progress is not None:
            progress(share * done / total, 1.0)

    turns = sherpa.sherpa_turns(audio, settings, progress=sherpa_progress)
    if model is None or not turns:
        if progress is not None:
            progress(1.0, 1.0)
        return turns

    def embed_progress(done: float, total: float) -> None:
        if progress is not None:
            progress(share + (1.0 - share) * done / total, 1.0)

    embeddings = sherpa.embed_turns(audio, turns, model, settings, progress=embed_progress)
    return clustering.recluster(
        turns,
        embeddings,
        threshold=settings.recluster_threshold,
        anchor_seconds=settings.recluster_anchor_seconds,
        join_similarity=settings.recluster_join_similarity,
        fold_seconds=settings.recluster_fold_seconds,
        fold_similarity=settings.recluster_fold_similarity,
    )
