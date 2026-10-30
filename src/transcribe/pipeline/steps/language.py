"""Language detection with faster-whisper, from the first speech of the recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transcribe.domain import Language
from transcribe.pipeline.offline import offline_imports
from transcribe.pipeline.steps.asr import whisper_ct2


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from transcribe.config import Settings


def detect_language(audio: npt.NDArray[np.float32], settings: Settings) -> Language | None:
    """Detect the spoken language from the first speech; None if the recording holds no speech."""
    offline_imports()
    opening = audio[: int(settings.language_detection_seconds * settings.sample_rate)]
    speech = whisper_ct2.speech(opening, settings)
    if not len(speech) and len(opening) < len(audio):
        speech = whisper_ct2.speech(audio, settings)  # silent opening: search the whole recording
    if not len(speech):
        return None
    model = whisper_ct2.load_model(settings.whisper_model_dir, settings)
    code, probability, _ = model.detect_language(
        speech,
        language_detection_segments=settings.language_detection_segments,
        language_detection_threshold=settings.language_detection_threshold,
    )
    return Language(code, float(probability))
