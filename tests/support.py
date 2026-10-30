"""Helpers shared by the tests: CLI runner, synthetic audio, fake engines."""

from __future__ import annotations

import sys
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from click.testing import CliRunner

from transcribe import application
from transcribe.cli import main
from transcribe.domain import Language, Turn, Word


if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from transcribe.events import ProgressCallback
    from transcribe.settings import Settings

SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class Result:
    exit_code: int
    stdout: str
    stderr: str


def run_main(*args: str) -> Result:
    """Run the console-script entry point (with its exit-code mapping) in isolation."""
    runner = CliRunner()
    exit_code: int = 0
    with runner.isolation() as (stdout, stderr, _):
        saved_argv = sys.argv
        sys.argv = ["transcribe", *args]
        try:
            main()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1
        finally:
            sys.argv = saved_argv
        return Result(exit_code, stdout.getvalue().decode(), stderr.getvalue().decode())


def write_wav(path: Path, samples: npt.NDArray[np.float32]) -> Path:
    """Write 16-bit mono PCM with the standard library only."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return path


def tone(seconds: float, frequency: float = 440.0) -> npt.NDArray[np.float32]:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


RECOGNIZED = "Confidential words"


@dataclass
class FakeEngines:
    """Stand-ins for the decoder and both engines, recording how they were called."""

    seconds: float = 12.0
    words: list[Word] = field(
        default_factory=lambda: [
            Word(0.5, 1.0, f" {RECOGNIZED}"),
            Word(1.0, 1.5, " here."),
            Word(3.0, 3.5, " Reply"),
            Word(9.0, 9.5, " orphan"),
        ]
    )
    turns: list[Turn] = field(default_factory=lambda: [Turn(0.0, 2.0, 7), Turn(2.5, 4.0, 3)])
    language: Language | None = field(default_factory=lambda: Language("en", 0.98))
    transcribed_in: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    fail_in: str | None = None
    failure: BaseException = field(default_factory=lambda: RuntimeError(f"engine says {RECOGNIZED}"))
    calls: list[str] = field(default_factory=list)

    def _maybe_fail(self, stage: str) -> None:
        self.calls.append(stage)
        if self.fail_in == stage:
            raise self.failure

    def preflight(self, settings: Settings) -> None:
        self._maybe_fail("preflight")

    def decoder(self, path: Path, settings: Settings) -> npt.NDArray[np.float32]:
        self._maybe_fail("decode")
        return np.zeros(int(self.seconds * settings.sample_rate), dtype=np.float32)

    def detector(self, audio: npt.NDArray[np.float32], settings: Settings) -> Language | None:
        self._maybe_fail("language detection")
        return self.language

    def transcriber(
        self, audio: npt.NDArray[np.float32], settings: Settings, language: str, *, progress: ProgressCallback
    ) -> list[Word]:
        progress(self.seconds / 2, self.seconds)
        self._maybe_fail("asr")
        progress(self.seconds, self.seconds)
        self.transcribed_in.append(language)
        self.models_used.append(settings.whisper_model_dir_for(language).name)
        return self.words

    def diarizer(self, audio: npt.NDArray[np.float32], settings: Settings, *, progress: ProgressCallback) -> list[Turn]:
        self._maybe_fail("diarization")
        progress(4, 4)
        return self.turns

    def engines(self) -> application.Engines:
        return application.Engines(
            preflight=self.preflight,
            decoder=self.decoder,
            detector=self.detector,
            transcriber=self.transcriber,
            diarizer=self.diarizer,
        )
