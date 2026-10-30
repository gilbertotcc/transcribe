"""What the pipeline reports while it runs; interfaces such as the CLI decide how to show it."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

    from transcribe.domain import Language

ProgressCallback = Callable[[float, float], None]
"""Called with `(done, total)` in the stage's own unit (audio seconds, chunks)."""


class Observer(Protocol):
    """Receives pipeline events; implementations must never show recognized text."""

    def stage_started(self, name: str) -> None: ...

    def stage_finished(self, name: str, seconds: float) -> None: ...

    def language(self, language: Language | None, *, threshold: float) -> None:
        """The language to transcribe in; None when no speech was found for detection."""

    def track(self, stage: str) -> AbstractContextManager[ProgressCallback]:
        """Progress of a long stage (ASR, diarization)."""

    def published(self, path: Path) -> None:
        """The transcript was written to `path`."""


class SilentObserver:
    """Ignores every event."""

    def stage_started(self, name: str) -> None:
        pass

    def stage_finished(self, name: str, seconds: float) -> None:
        pass

    def language(self, language: Language | None, *, threshold: float) -> None:
        pass

    @contextmanager
    def track(self, stage: str) -> Iterator[ProgressCallback]:
        yield lambda _done, _total: None

    def published(self, path: Path) -> None:
        pass
