from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from support import RECOGNIZED, run_main


if TYPE_CHECKING:
    from pathlib import Path

    from support import FakeEngines

EXPECTED = """\
# Transcript: meeting.wav

> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted.

[00:00:00] SPEAKER_01: Confidential words here.

[00:00:03] SPEAKER_02: Reply

[00:00:09] UNKNOWN: orphan
"""


def test_stdout_mode(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert result.stdout == EXPECTED
    assert fakes.calls == ["preflight", "decode", "asr", "diarization"]
    assert "transcribe: asr done in" in result.stderr
    assert RECOGNIZED not in result.stderr


def test_file_mode(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    destination = tmp_path / "transcript.md"

    result = run_main(str(audio_file), "--output", str(destination))

    assert result.exit_code == 0
    assert result.stdout == ""
    assert destination.read_text(encoding="utf-8") == EXPECTED
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meeting.wav", "transcript.md"]


def test_no_speech(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.words = []

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert result.stdout.endswith("Not redacted.\n\nNo speech detected.\n")
    assert "diarization" not in fakes.calls


def test_too_short_audio_skips_engines(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.seconds = 0.4

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert result.stdout.endswith("No speech detected.\n")
    assert fakes.calls == ["preflight", "decode"]


def test_no_turns_gives_unknown(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.turns = []

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert "[00:00:00] UNKNOWN: Confidential words here." in result.stdout
    assert "SPEAKER_" not in result.stdout


@pytest.mark.parametrize("stage", ["decode", "asr", "diarization"])
def test_engine_failure_is_exit_1_without_text(
    audio_file: Path, tmp_path: Path, fakes: FakeEngines, stage: str
) -> None:
    fakes.fail_in = stage
    destination = tmp_path / "transcript.md"

    result = run_main(str(audio_file), "--output", str(destination))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert f"transcribe: {stage} failed (RuntimeError)" in result.stderr
    assert RECOGNIZED not in result.stderr
    assert "Traceback" not in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meeting.wav"]


def test_missing_models_is_exit_2(audio_file: Path, fakes: FakeEngines) -> None:
    from transcribe.errors import MissingModelsError

    fakes.fail_in = "preflight"
    fakes.failure = MissingModelsError("missing model files")

    result = run_main(str(audio_file))

    assert result.exit_code == 2
    assert "missing model files" in result.stderr
    assert fakes.calls == ["preflight"]


def test_keyboard_interrupt_is_exit_130(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    fakes.fail_in = "asr"
    fakes.failure = KeyboardInterrupt()
    destination = tmp_path / "transcript.md"

    result = run_main(str(audio_file), "--output", str(destination))

    assert result.exit_code == 130
    assert result.stdout == ""
    assert "interrupted" in result.stderr
    assert not destination.exists()
