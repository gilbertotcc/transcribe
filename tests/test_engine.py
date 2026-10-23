"""Opt-in tests with the real engines: `uv run pytest -m engine`.

They need provisioned models and a short public-domain English clip with two
speakers (never committed). Set TRANSCRIBE_TEST_CLIP, or place it at
bench/clips/dialogue30.wav; MP3/M4A variants next to it are used when present.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from support import write_wav

from transcribe.application import build_transcript
from transcribe.audio import decode
from transcribe.models import required_paths
from transcribe.settings import Settings


pytestmark = pytest.mark.engine

CLIP = Path(os.environ.get("TRANSCRIBE_TEST_CLIP", Path(__file__).parent.parent / "bench/clips/dialogue30.wav"))
MARKDOWN_LINE = re.compile(
    r"^(# Transcript: .+|> Automatic transcript\..*|\[\d{2,}:\d{2}:\d{2}\] (SPEAKER_\d{2,}|UNKNOWN): .+|)$"
)


@pytest.fixture(scope="module", autouse=True)
def _require_assets() -> None:
    if not all(path.is_file() for path in required_paths(Settings())):
        pytest.skip("models not provisioned (run scripts/provision_models.py)")
    if not CLIP.is_file():
        pytest.skip(f"test clip not found: {CLIP}")


def test_words_have_monotonic_plausible_times_and_two_speakers() -> None:
    transcript = build_transcript(CLIP, Settings())

    starts = [item.word.start for item in transcript.words]
    assert len(starts) > 20
    assert starts == sorted(starts)
    assert all(0 <= item.word.start <= item.word.end <= transcript.audio_seconds for item in transcript.words)
    assert transcript.speakers_found >= 2


def test_leading_silence_keeps_original_timestamps(tmp_path: Path) -> None:
    settings = Settings()
    audio = decode(CLIP, settings)
    padded = write_wav(tmp_path / "padded.wav", np.concatenate([np.zeros(10 * 16_000, np.float32), audio]))

    plain = build_transcript(CLIP, settings).words
    shifted = build_transcript(padded, settings).words

    assert shifted[0].word.start == pytest.approx(plain[0].word.start + 10.0, abs=0.5)
    assert shifted[-1].word.end == pytest.approx(plain[-1].word.end + 10.0, abs=0.5)


@pytest.mark.parametrize("suffix", [".wav", ".mp3", ".m4a"])
def test_cli_stdout_is_markdown_only(suffix: str) -> None:
    clip = CLIP.with_suffix(suffix)
    if not clip.is_file():
        pytest.skip(f"no {suffix} variant of the test clip")

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and arguments
        [sys.executable, "-m", "transcribe", str(clip)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.split("\n")
    assert lines[0] == f"# Transcript: {clip.name}"
    assert completed.stdout.endswith("\n")
    assert all(MARKDOWN_LINE.match(line) for line in lines), completed.stdout
    assert re.search(r"^\[\d{2}:\d{2}:\d{2}\] SPEAKER_01: ", completed.stdout, re.MULTILINE)
