"""Pipeline orchestration. Engines are parameters so tests can pass fakes."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from transcribe.domain import Language
from transcribe.errors import AttributionError, DecodeError, InferenceError, TranscribeError
from transcribe.pipeline.events import SilentObserver
from transcribe.pipeline.steps import attribution, output
from transcribe.pipeline.steps.rendering import render_markdown


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from transcribe.config import Settings
    from transcribe.domain import AttributedWord, Paragraph, Turn, Word
    from transcribe.pipeline.events import Observer, ProgressCallback

    AudioArray = npt.NDArray[np.float32]


class Decoder(Protocol):
    def __call__(self, path: Path, settings: Settings) -> AudioArray: ...


class Detector(Protocol):
    def __call__(self, audio: AudioArray, settings: Settings) -> Language | None: ...


class Transcriber(Protocol):
    def __call__(
        self, audio: AudioArray, settings: Settings, language: str, *, progress: ProgressCallback
    ) -> list[Word]: ...


class Diarizer(Protocol):
    def __call__(self, audio: AudioArray, settings: Settings, *, progress: ProgressCallback) -> list[Turn]: ...


@dataclass(frozen=True, slots=True)
class Engines:
    """Pipeline dependencies; `None` selects the real implementation (imported lazily)."""

    preflight: Callable[[Settings], None] | None = None
    decoder: Decoder | None = None
    detector: Detector | None = None
    transcriber: Transcriber | None = None
    diarizer: Diarizer | None = None


@dataclass(slots=True)
class Transcript:
    """Everything one run produced; holds recognized text, so never log or persist it."""

    document: str
    audio_seconds: float
    words: list[AttributedWord]
    paragraphs: list[Paragraph]
    speakers_found: int
    language: Language | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)


@contextmanager
def _stage(name: str, timings: dict[str, float], error: type[TranscribeError], observer: Observer) -> Iterator[None]:
    """Time a stage, report it to `observer`, and turn engine exceptions into `error`."""
    observer.stage_started(name)
    started = time.perf_counter()
    try:
        yield
    except TranscribeError:
        raise
    except Exception as exc:
        # The engine message may quote recognized text, so only the type is reported.
        msg = f"{name} failed ({type(exc).__name__})"
        raise error(msg) from exc
    elapsed = time.perf_counter() - started
    timings[name] = elapsed
    observer.stage_finished(name, elapsed)


def build_transcript(
    audio_file: Path, settings: Settings, engines: Engines | None = None, *, observer: Observer | None = None
) -> Transcript:
    """Run every stage up to the rendered Markdown document, reporting stages and progress to `observer`."""
    engines = engines or Engines()
    observer = observer or SilentObserver()
    preflight = engines.preflight or _default_preflight
    decoder = engines.decoder or _default_decoder
    detector = engines.detector or _default_detector
    transcriber = engines.transcriber or _default_transcriber
    diarizer = engines.diarizer or _default_diarizer

    preflight(settings)
    timings: dict[str, float] = {}
    with _stage("decode", timings, DecodeError, observer):
        audio = decoder(audio_file, settings)
    audio_seconds = len(audio) / settings.sample_rate

    words: list[Word] = []
    turns: list[Turn] = []
    language = None if settings.language is None else Language(settings.language, None)
    if audio_seconds >= settings.min_audio_seconds:
        if language is None:
            with _stage("language detection", timings, InferenceError, observer):
                language = detector(audio, settings)
        observer.language(language, threshold=settings.language_detection_threshold)
        if language is not None:  # None: no speech anywhere
            with _stage("asr", timings, InferenceError, observer), observer.track("asr") as update:
                words = transcriber(audio, settings, language.code, progress=update)
        if words:
            with _stage("diarization", timings, InferenceError, observer), observer.track("diarization") as update:
                turns = diarizer(audio, settings, progress=update)
    del audio  # up to ~230 MB per hour of audio

    with _stage("render", timings, AttributionError, observer):
        attributed = attribution.attribute(words, turns)
        paragraphs = attribution.paragraphs(
            attributed,
            max_gap=settings.paragraph_gap_seconds,
            max_duration=settings.paragraph_max_seconds,
        )
        document = render_markdown(audio_file.name, paragraphs, language)
    return Transcript(
        document=document,
        audio_seconds=audio_seconds,
        words=attributed,
        paragraphs=paragraphs,
        speakers_found=len({turn.speaker for turn in turns}),
        language=language,
        stage_seconds=timings,
    )


def run(
    audio_file: Path,
    output_path: Path | None,
    settings: Settings,
    engines: Engines | None = None,
    *,
    observer: Observer | None = None,
) -> None:
    """Transcribe `audio_file` and publish the Markdown to `output_path` or stdout."""
    observer = observer or SilentObserver()
    if output_path is not None:
        output.validate_output_path(output_path)
    transcript = build_transcript(audio_file, settings, engines, observer=observer)
    if output_path is None:
        output.write_stdout(transcript.document)
    else:
        output.publish_file(transcript.document, output_path)
        observer.published(output_path)


def _default_preflight(settings: Settings) -> None:
    from transcribe.models import check_assets  # keep startup light

    check_assets(settings)


def _default_decoder(path: Path, settings: Settings) -> AudioArray:
    from transcribe.pipeline.steps.decode import decode  # engines load lazily

    return decode(path, settings)


def _default_detector(audio: AudioArray, settings: Settings) -> Language | None:
    from transcribe.pipeline.steps.language import detect_language  # engines load lazily

    return detect_language(audio, settings)


def _default_transcriber(
    audio: AudioArray, settings: Settings, language: str, *, progress: ProgressCallback
) -> list[Word]:
    from transcribe.pipeline.steps.asr import transcribe  # engines load lazily

    return transcribe(audio, settings, language, progress=progress)


def _default_diarizer(audio: AudioArray, settings: Settings, *, progress: ProgressCallback) -> list[Turn]:
    from transcribe.pipeline.steps.diarization import diarize  # engines load lazily

    return diarize(audio, settings, progress=progress)
