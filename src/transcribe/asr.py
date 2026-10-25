"""faster-whisper adapter: language detection, and audio array to timed words."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from transcribe.audio import offline_imports
from transcribe.domain import Language, Word, clamp_interval


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from transcribe.settings import Settings

_PROGRESS_EVERY_SECONDS = 300.0


def _load_model(model_dir: Path, settings: Settings) -> Any:  # an untyped faster_whisper.WhisperModel
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


def _speech(audio: npt.NDArray[np.float32], settings: Settings) -> npt.NDArray[np.float32]:
    """The detected speech of `audio`, concatenated (all of `audio` when the VAD is off)."""
    if not settings.vad_filter:
        return audio
    import numpy as np
    from faster_whisper.vad import collect_chunks, get_speech_timestamps

    chunks, _ = collect_chunks(audio, get_speech_timestamps(audio), sampling_rate=settings.sample_rate)
    return np.concatenate(chunks)


def detect_language(audio: npt.NDArray[np.float32], settings: Settings) -> Language | None:
    """Detect the spoken language from the first speech; None if the recording holds no speech."""
    offline_imports()
    opening = audio[: int(settings.language_detection_seconds * settings.sample_rate)]
    speech = _speech(opening, settings)
    if not len(speech) and len(opening) < len(audio):
        speech = _speech(audio, settings)  # silent opening: search the whole recording
    if not len(speech):
        return None
    model = _load_model(settings.whisper_model_dir, settings)
    code, probability, _ = model.detect_language(
        speech,
        language_detection_segments=settings.language_detection_segments,
        language_detection_threshold=settings.language_detection_threshold,
    )
    return Language(code, float(probability))


def transcribe(audio: npt.NDArray[np.float32], settings: Settings, language: str) -> list[Word]:
    from faster_whisper import BatchedInferencePipeline  # engines load lazily

    model = _load_model(settings.whisper_model_dir_for(language), settings)
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
    next_report = _PROGRESS_EVERY_SECONDS
    for segment in segments:  # a lazy generator: inference happens while iterating
        for word in segment.words or ():
            start, end = clamp_interval(word.start, word.end, duration)
            words.append(Word(start, end, word.word))
        if segment.end >= next_report:
            click.echo(f"transcribe: asr at {segment.end / 60:.0f} of {duration / 60:.0f} min", err=True)
            next_report = segment.end + _PROGRESS_EVERY_SECONDS
    return words
