from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from support import tone, write_wav

from transcribe.audio import decode
from transcribe.errors import DecodeError
from transcribe.settings import Settings


if TYPE_CHECKING:
    from pathlib import Path


def test_decode_wav_to_16k_mono_float32(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "tone.wav", tone(1.5))
    before = path.read_bytes()

    audio = decode(path, Settings())

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 24_000) <= 160
    assert 0.2 < float(np.abs(audio).max()) <= 0.31
    assert path.read_bytes() == before  # source never modified


def test_corrupt_file_is_a_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.m4a"
    path.write_bytes(b"\x00\x01not really audio" * 64)

    with pytest.raises(DecodeError, match=r"cannot decode audio \(\w+\)"):
        decode(path, Settings())
