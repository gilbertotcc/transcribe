"""faster-whisper adapter: audio array to timed words."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from transcribe.audio import offline_imports
from transcribe.domain import Word, clamp_interval


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from transcribe.settings import Settings

_PROGRESS_EVERY_SECONDS = 300.0


def transcribe(audio: npt.NDArray[np.float32], settings: Settings) -> list[Word]:
    offline_imports()
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # engines load lazily

    # A directory path (never a model name) plus local_files_only: nothing can be downloaded.
    model = WhisperModel(
        str(settings.whisper_model_dir),
        device="cpu",
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        local_files_only=True,
    )
    options: dict[str, Any] = {
        "language": "en",
        "beam_size": settings.beam_size,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "vad_filter": settings.vad_filter,
    }
    if settings.asr_batch_size > 0:
        segments, _ = BatchedInferencePipeline(model).transcribe(audio, batch_size=settings.asr_batch_size, **options)
    else:
        segments, _ = model.transcribe(audio, **options)

    duration = len(audio) / settings.sample_rate
    words: list[Word] = []
    next_report = _PROGRESS_EVERY_SECONDS
    for segment in segments:  # a lazy generator: inference happens while iterating
        for word in segment.words or ():
            start, end = clamp_interval(word.start, word.end, duration)
            words.append(Word(start, end, word.word))
        if segment.end >= next_report:
            click.echo(f"transcribe: asr at {segment.end / 60:.0f} of {duration / 60:.0f} min", err=True)
            next_report = segment.end + _PROGRESS_EVERY_SECONDS
    return words
