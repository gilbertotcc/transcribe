"""Pipeline orchestration. Engines are parameters so tests can pass fakes."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import click

from transcribe import attribution, output
from transcribe.errors import AttributionError, DecodeError, InferenceError, TranscribeError
from transcribe.rendering import render_markdown


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from transcribe.domain import AttributedWord, Paragraph, Turn, Word
    from transcribe.settings import Settings

    AudioArray = npt.NDArray[np.float32]


class Decoder(Protocol):
    def __call__(self, path: Path, settings: Settings) -> AudioArray: ...


class Transcriber(Protocol):
    def __call__(self, audio: AudioArray, settings: Settings) -> list[Word]: ...


class Diarizer(Protocol):
    def __call__(self, audio: AudioArray, settings: Settings) -> list[Turn]: ...


@dataclass(frozen=True, slots=True)
class Engines:
    """Pipeline dependencies; `None` selects the real implementation (imported lazily)."""

    preflight: Callable[[Settings], None] | None = None
    decoder: Decoder | None = None
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
    stage_seconds: dict[str, float] = field(default_factory=dict)


@contextmanager
def _stage(name: str, timings: dict[str, float], error: type[TranscribeError]) -> Iterator[None]:
    """Time a stage, report it on stderr, and turn engine exceptions into `error`."""
    click.echo(f"transcribe: {name}...", err=True)
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
    click.echo(f"transcribe: {name} done in {elapsed:.1f} s", err=True)


def build_transcript(audio_file: Path, settings: Settings, engines: Engines | None = None) -> Transcript:
    """Run every stage up to the rendered Markdown document."""
    engines = engines or Engines()
    preflight = engines.preflight or _default_preflight
    decoder = engines.decoder or _default_decoder
    transcriber = engines.transcriber or _default_transcriber
    diarizer = engines.diarizer or _default_diarizer

    preflight(settings)
    timings: dict[str, float] = {}
    with _stage("decode", timings, DecodeError):
        audio = decoder(audio_file, settings)
    audio_seconds = len(audio) / settings.sample_rate

    words: list[Word] = []
    turns: list[Turn] = []
    if audio_seconds >= settings.min_audio_seconds:
        with _stage("asr", timings, InferenceError):
            words = transcriber(audio, settings)
        if words:
            with _stage("diarization", timings, InferenceError):
                turns = diarizer(audio, settings)
    del audio  # up to ~230 MB per hour of audio

    with _stage("render", timings, AttributionError):
        attributed = attribution.attribute(words, turns)
        paragraphs = attribution.paragraphs(
            attributed,
            max_gap=settings.paragraph_gap_seconds,
            max_duration=settings.paragraph_max_seconds,
        )
        document = render_markdown(audio_file.name, paragraphs)
    return Transcript(
        document=document,
        audio_seconds=audio_seconds,
        words=attributed,
        paragraphs=paragraphs,
        speakers_found=len({turn.speaker for turn in turns}),
        stage_seconds=timings,
    )


def run(audio_file: Path, output_path: Path | None, settings: Settings, engines: Engines | None = None) -> None:
    """Transcribe `audio_file` and publish the Markdown to `output_path` or stdout."""
    if output_path is not None:
        output.validate_output_path(output_path)
    transcript = build_transcript(audio_file, settings, engines)
    if output_path is None:
        output.write_stdout(transcript.document)
    else:
        output.publish_file(transcript.document, output_path)
        click.echo(f"transcribe: wrote {output_path}", err=True)


def _default_preflight(settings: Settings) -> None:
    from transcribe.models import check_assets  # keep startup light

    check_assets(settings)


def _default_decoder(path: Path, settings: Settings) -> AudioArray:
    from transcribe.audio import decode  # engines load lazily

    return decode(path, settings)


def _default_transcriber(audio: AudioArray, settings: Settings) -> list[Word]:
    from transcribe.asr import transcribe  # engines load lazily

    return transcribe(audio, settings)


def _default_diarizer(audio: AudioArray, settings: Settings) -> list[Turn]:
    from transcribe.diarization import diarize  # engines load lazily

    return diarize(audio, settings)
