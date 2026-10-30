"""Progress of the long stages (ASR, diarization) on stderr: a bar with ETA, plain lines, or nothing."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

import click


if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from typing import TextIO

    from transcribe.events import ProgressCallback

_BAR_STEPS = 1000  # bar resolution; Click counts integer steps


class Reporter(Protocol):
    def track(self, stage: str) -> AbstractContextManager[ProgressCallback]: ...


class NullReporter:
    """Shows nothing (`--no-progress`)."""

    @contextmanager
    def track(self, stage: str) -> Iterator[ProgressCallback]:
        yield lambda _done, _total: None


class LineReporter:
    """One stderr line per 10 % of a stage; readable in logs and redirected output."""

    @contextmanager
    def track(self, stage: str) -> Iterator[ProgressCallback]:
        last_decile = 0

        def update(done: float, total: float) -> None:
            nonlocal last_decile
            decile = min(int(done * 10 // total), 10) if total > 0 else 0
            if decile > last_decile:
                last_decile = decile
                click.echo(f"transcribe: {stage} {decile * 10}%", err=True)

        yield update


class BarReporter:
    """A live bar with percentage and estimated time left, for an interactive terminal."""

    def __init__(self, file: TextIO | None = None) -> None:
        self._file = file

    @contextmanager
    def track(self, stage: str) -> Iterator[ProgressCallback]:
        file = self._file or sys.stderr
        bar = click.progressbar(length=_BAR_STEPS, label=f"transcribe: {stage}", show_eta=True, file=file)
        shown = 0

        def update(done: float, total: float) -> None:
            nonlocal shown
            step = min(int(done * _BAR_STEPS / total), _BAR_STEPS) if total > 0 else 0
            if step > shown:
                bar.update(step - shown)
                shown = step

        with bar:  # always ends the bar's line, also on errors and Ctrl-C
            yield update


def reporter(*, enabled: bool) -> Reporter:
    """The reporter for the CLI: a bar on a terminal, lines otherwise, nothing if disabled."""
    if not enabled:
        return NullReporter()
    if sys.stderr.isatty():
        return BarReporter()
    return LineReporter()
