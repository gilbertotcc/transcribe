"""sherpa-onnx adapter: audio array to anonymous speaker turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transcribe.domain import Turn, clamp_interval
from transcribe.errors import InferenceError


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from transcribe.progress import ProgressCallback
    from transcribe.settings import Settings


def diarize(
    audio: npt.NDArray[np.float32], settings: Settings, *, progress: ProgressCallback | None = None
) -> list[Turn]:
    import sherpa_onnx  # engines load lazily

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(settings.segmentation_model_path),
                window_shift_ratio=settings.window_shift_ratio,
            ),
            num_threads=settings.diarization_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(settings.embedding_model_path),
            num_threads=settings.diarization_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=settings.clustering_threshold),
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
