from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from transcribe.config import Settings
from transcribe.pipeline.steps import asr
from transcribe.pipeline.steps.asr import whisper_ct2, whisper_mlx


if TYPE_CHECKING:
    from collections.abc import Callable

    from transcribe.domain import Word


@pytest.mark.parametrize(("backend", "expected"), [("mlx", "whisper_mlx"), ("ctranslate2", "whisper_ct2")])
def test_transcribe_follows_the_backend(monkeypatch: pytest.MonkeyPatch, backend: str, expected: str) -> None:
    used: list[str] = []

    def fake(name: str) -> Callable[..., list[Word]]:
        def transcribe(*_args: object, **_kwargs: object) -> list[Word]:
            used.append(name)
            return []

        return transcribe

    monkeypatch.setattr(whisper_mlx, "transcribe", fake("whisper_mlx"))
    monkeypatch.setattr(whisper_ct2, "transcribe", fake("whisper_ct2"))
    settings = Settings(asr_backend=backend, mlx_whisper_model="mlx-whisper-large-v3")

    asr.transcribe(np.zeros(16, np.float32), settings, "en", progress=lambda _d, _t: None)

    assert used == [expected]
