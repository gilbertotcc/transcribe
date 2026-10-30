from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
from support import FakeEngines, tone, write_wav

from transcribe import application, cli
from transcribe.hardware import Host


if TYPE_CHECKING:
    from pathlib import Path

    from transcribe.events import Observer
    from transcribe.settings import Settings


INTEL_HOST = Host("intel-mac", "Intel(R) Core(TM) i5-1038NG7", 16 * 2**30)


@pytest.fixture(autouse=True)
def host(monkeypatch: pytest.MonkeyPatch) -> Host:
    """CLI tests behave the same on every machine: the CLI sees the 2020 Intel reference host."""
    monkeypatch.setattr(cli, "detect_host", lambda: INTEL_HOST)
    return INTEL_HOST


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "meeting.wav", tone(1.0))


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeEngines:
    """Make the CLI run the real application with fake engines."""
    engines = FakeEngines()
    real_run = application.run

    def run(audio_file: Path, output_path: Path | None, settings: Settings, *, observer: Observer) -> None:
        real_run(audio_file, output_path, settings, engines.engines(), observer=observer)

    monkeypatch.setattr(application, "run", functools.wraps(real_run)(run))
    return engines
