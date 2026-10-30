"""Speech recognition: audio array to timed words, on the backend `Settings.asr_backend` selects."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from transcribe.config import Settings
    from transcribe.domain import Word
    from transcribe.pipeline.events import ProgressCallback


def transcribe(
    audio: npt.NDArray[np.float32], settings: Settings, language: str, *, progress: ProgressCallback | None = None
) -> list[Word]:
    if settings.uses_mlx:
        from transcribe.pipeline.steps.asr import whisper_mlx  # engines load lazily

        return whisper_mlx.transcribe(audio, settings, language, progress=progress)
    from transcribe.pipeline.steps.asr import whisper_ct2  # engines load lazily

    return whisper_ct2.transcribe(audio, settings, language, progress=progress)
