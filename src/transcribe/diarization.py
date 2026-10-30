"""sherpa-onnx adapter: audio array to anonymous speaker turns, re-clustered from per-turn embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from transcribe import clustering
from transcribe.domain import Turn, clamp_interval
from transcribe.errors import InferenceError


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from transcribe.events import ProgressCallback
    from transcribe.settings import Settings

EDGE_SECONDS = 0.1  # trimmed from each end of a turn before embedding it, away from neighbouring speech
MIN_EMBED_SECONDS = 0.25  # shorter turns get no embedding
_SHERPA_SHARE = 0.8  # of the stage's progress, when re-clustering follows (it takes about a fifth of the time)


def diarize(
    audio: npt.NDArray[np.float32], settings: Settings, *, progress: ProgressCallback | None = None
) -> list[Turn]:
    model = settings.recluster_embedding_model_path
    share = 1.0 if model is None else _SHERPA_SHARE

    def sherpa_progress(done: float, total: float) -> None:
        if progress is not None:
            progress(share * done / total, 1.0)

    turns = sherpa_turns(audio, settings, progress=sherpa_progress)
    if model is None or not turns:
        if progress is not None:
            progress(1.0, 1.0)
        return turns

    def embed_progress(done: float, total: float) -> None:
        if progress is not None:
            progress(share + (1.0 - share) * done / total, 1.0)

    embeddings = embed_turns(audio, turns, model, settings, progress=embed_progress)
    return clustering.recluster(
        turns,
        embeddings,
        threshold=settings.recluster_threshold,
        anchor_seconds=settings.recluster_anchor_seconds,
        join_similarity=settings.recluster_join_similarity,
        fold_seconds=settings.recluster_fold_seconds,
        fold_similarity=settings.recluster_fold_similarity,
    )


def sherpa_turns(
    audio: npt.NDArray[np.float32],
    settings: Settings,
    *,
    num_clusters: int = -1,
    progress: ProgressCallback | None = None,
) -> list[Turn]:
    """sherpa-onnx diarization alone; `num_clusters` -1 infers the count (other values: diagnostics only)."""
    import sherpa_onnx  # engines load lazily

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(settings.segmentation_model_path),
                window_shift_ratio=settings.window_shift_ratio,
            ),
            num_threads=settings.diarization_threads,
            provider=settings.diarization_provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(settings.embedding_model_path),
            num_threads=settings.diarization_threads,
            provider=settings.diarization_provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=settings.clustering_threshold),
        min_duration_on=settings.min_duration_on,
        min_duration_off=settings.min_duration_off,
    )
    if not config.validate():
        msg = "diarization model configuration is invalid"
        raise InferenceError(msg)
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    if diarizer.sample_rate != settings.sample_rate:
        msg = f"diarization expects {diarizer.sample_rate} Hz audio, not {settings.sample_rate} Hz"
        raise InferenceError(msg)

    def callback(done: int, total: int) -> int:
        if progress is not None:
            progress(done, total)
        return 0  # non-zero would abort

    result = diarizer.process(audio, callback=callback).sort_by_start_time()
    duration = len(audio) / settings.sample_rate
    turns = []
    for segment in result:
        start, end = clamp_interval(segment.start, segment.end, duration)
        turns.append(Turn(start, end, int(segment.speaker)))
    return turns


def embed_turns(
    audio: npt.NDArray[np.float32],
    turns: Sequence[Turn],
    model: Path,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
) -> npt.NDArray[np.float64]:
    """One unit speaker embedding per turn, from inside its edges; zeros for turns too short to embed."""
    import sherpa_onnx  # engines load lazily

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model), num_threads=settings.diarization_threads, provider=settings.diarization_provider
    )
    if not config.validate():
        msg = "speaker embedding model configuration is invalid"
        raise InferenceError(msg)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    rate = settings.sample_rate
    vectors = np.zeros((len(turns), extractor.dim), dtype=np.float64)
    for index, turn in enumerate(turns):
        start, end = turn.start + EDGE_SECONDS, turn.end - EDGE_SECONDS
        if end - start < MIN_EMBED_SECONDS:
            start, end = turn.start, turn.end
        if end - start >= MIN_EMBED_SECONDS:
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=rate, waveform=audio[int(start * rate) : int(end * rate)])
            stream.input_finished()
            vectors[index] = np.array(extractor.compute(stream), dtype=np.float64)
        if progress is not None:
            progress(index + 1, len(turns))
    return clustering.normalize(vectors)
