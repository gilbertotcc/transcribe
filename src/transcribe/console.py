"""Pipeline events as `transcribe:` lines on stderr."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from transcribe.languages import label


if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

    from transcribe.domain import Language
    from transcribe.events import ProgressCallback
    from transcribe.progress import Reporter


class ConsoleObserver:
    """Reports stages, the language and the published file on stderr; `reporter` shows progress."""

    def __init__(self, reporter: Reporter) -> None:
        self._reporter = reporter

    def stage_started(self, name: str) -> None:
        click.echo(f"transcribe: {name}...", err=True)

    def stage_finished(self, name: str, seconds: float) -> None:
        click.echo(f"transcribe: {name} done in {seconds:.1f} s", err=True)

    def language(self, language: Language | None, *, threshold: float) -> None:
        if language is None:
            click.echo("transcribe: no speech found for language detection", err=True)
        elif language.probability is None:
            click.echo(f"transcribe: language {label(language.code)}, forced", err=True)
        else:
            click.echo(f"transcribe: language {label(language.code)}, p={language.probability:.2f}", err=True)
            if language.probability <= threshold:
                click.echo(
                    "transcribe: warning: language detection is uncertain; pass --language to force it", err=True
                )

    def track(self, stage: str) -> AbstractContextManager[ProgressCallback]:
        return self._reporter.track(stage)

    def published(self, path: Path) -> None:
        click.echo(f"transcribe: wrote {path}", err=True)
