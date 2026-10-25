from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
from support import FakeEngines, tone, write_wav

from transcribe import application


if TYPE_CHECKING:
    from pathlib import Path

    from transcribe.settings import Settings


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "meeting.wav", tone(1.0))


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeEngines:
    """Make the CLI run the real application with fake engines."""
    engines = FakeEngines()
    real_run = application.run

    def run(audio_file: Path, output_path: Path | None, settings: Settings, *, show_progress: bool) -> None:
        real_run(audio_file, output_path, settings, engines.engines(), show_progress=show_progress)

    monkeypatch.setattr(application, "run", functools.wraps(real_run)(run))
    return engines
