"""Decode any container PyAV understands into one 16 kHz mono float32 array."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from transcribe.errors import DecodeError


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from transcribe.settings import Settings


def offline_imports() -> None:
    """Forbid Hugging Face Hub network access before faster-whisper is imported."""
    os.environ["HF_HUB_OFFLINE"] = "1"


def decode(path: Path, settings: Settings) -> npt.NDArray[np.float32]:
    """Decode the first audio stream; the source file is only ever opened for reading."""
    offline_imports()
    import av  # engines load lazily
    from faster_whisper import decode_audio

    try:
        with path.open("rb") as stream:
            audio: npt.NDArray[np.float32] = decode_audio(stream, sampling_rate=settings.sample_rate)
    except (av.error.FFmpegError, IndexError, StopIteration) as exc:
        # IndexError/StopIteration: the container has no audio stream.
        msg = f"cannot decode audio ({type(exc).__name__})"
        raise DecodeError(msg) from exc
    return audio
