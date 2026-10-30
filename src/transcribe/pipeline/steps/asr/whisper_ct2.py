"""faster-whisper adapter: audio array to timed words on the CPU; the model loader and VAD."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transcribe.domain import Word, clamp_interval
from transcribe.pipeline.offline import offline_imports


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from transcribe.config import Settings
    from transcribe.pipeline.events import ProgressCallback


def load_model(model_dir: Path, settings: Settings) -> Any:  # an untyped faster_whisper.WhisperModel
    offline_imports()
    from faster_whisper import WhisperModel  # engines load lazily

    # A directory path (never a model name) plus local_files_only: nothing can be downloaded.
    return WhisperModel(
        str(model_dir),
        device="cpu",
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        local_files_only=True,
    )


def speech(audio: npt.NDArray[np.float32], settings: Settings) -> npt.NDArray[np.float32]:
    """The detected speech of `audio`, concatenated (all of `audio` when the VAD is off)."""
    if not settings.vad_filter:
        return audio
    import numpy as np
    from faster_whisper.vad import collect_chunks, get_speech_timestamps

    chunks, _ = collect_chunks(audio, get_speech_timestamps(audio), sampling_rate=settings.sample_rate)
    return np.concatenate(chunks)


def transcribe(
    audio: npt.NDArray[np.float32], settings: Settings, language: str, *, progress: ProgressCallback | None = None
) -> list[Word]:
    from faster_whisper import BatchedInferencePipeline  # engines load lazily

    model = load_model(settings.whisper_model_dir_for(language), settings)
    options: dict[str, Any] = {
        "language": language,
        "beam_size": settings.beam_size,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "word_timestamps": True,
        "vad_filter": settings.vad_filter,
    }
    if settings.asr_batch_size > 0:
        segments, _ = BatchedInferencePipeline(model).transcribe(audio, batch_size=settings.asr_batch_size, **options)
    else:
        segments, _ = model.transcribe(audio, **options)

    duration = len(audio) / settings.sample_rate
    words: list[Word] = []
    for segment in segments:  # a lazy generator: inference happens while iterating
        for word in segment.words or ():
            start, end = clamp_interval(word.start, word.end, duration)
            words.append(Word(start, end, word.word))
        if progress is not None:
            progress(segment.end, duration)
    if progress is not None:
        progress(duration, duration)  # trailing silence yields no segment
    return words
