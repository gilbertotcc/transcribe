from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from support import RECOGNIZED, run_main

from transcribe import application, cli, hardware
from transcribe.domain import Language, Word
from transcribe.hardware import Host
from transcribe.settings import Settings


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from support import FakeEngines

EXPECTED = """\
# Transcript: meeting.wav

Language: English (en), detected

> Automatic transcript. Speaker labels are inferred. Review before use. Not redacted.

[00:00:00] SPEAKER_01: Confidential words here.

[00:00:03] SPEAKER_02: Reply

[00:00:09] UNKNOWN: orphan
"""


def test_stdout_mode(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert result.stdout == EXPECTED
    assert fakes.calls == ["preflight", "decode", "language detection", "asr", "diarization"]
    assert fakes.transcribed_in == ["en"]
    assert "transcribe: asr done in" in result.stderr
    assert "transcribe: asr 50%" in result.stderr  # stderr is no terminal here: plain progress lines
    assert "transcribe: asr 100%" in result.stderr
    assert "transcribe: diarization 100%" in result.stderr
    assert "transcribe: language English (en), p=0.98" in result.stderr
    assert "warning" not in result.stderr
    assert RECOGNIZED not in result.stderr


def test_no_progress_hides_progress_lines(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--no-progress")

    assert result.exit_code == 0
    assert result.stdout == EXPECTED
    assert "%" not in result.stderr
    assert "transcribe: asr done in" in result.stderr


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
    assert "Language: English (en), detected" in result.stdout
    assert "diarization" not in fakes.calls


def test_detected_language_is_transcribed_and_shown(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.language = Language("it", 0.91)

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert fakes.transcribed_in == ["it"]
    assert "\n\nLanguage: Italian (it), detected\n\n" in result.stdout
    assert "transcribe: language Italian (it), p=0.91" in result.stderr


@pytest.mark.parametrize("given", ["it", "IT", " It "])
def test_forced_language_skips_detection(audio_file: Path, fakes: FakeEngines, given: str) -> None:
    result = run_main(str(audio_file), "--language", given)

    assert result.exit_code == 0
    assert "language detection" not in fakes.calls
    assert fakes.transcribed_in == ["it"]
    assert "\n\nLanguage: Italian (it), forced\n\n" in result.stdout
    assert "transcribe: language Italian (it), forced" in result.stderr


def test_unknown_language_is_exit_2_before_preflight(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--language", "xx")

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Invalid value for '--language'" in result.stderr
    assert "unknown language code 'xx'" in result.stderr
    assert fakes.calls == []


def test_uncertain_language_warns_and_continues(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.language = Language("it", 0.41)

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert fakes.transcribed_in == ["it"]
    assert "transcribe: warning: language detection is uncertain; pass --language to force it" in result.stderr


def test_no_speech_for_detection_skips_asr(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.language = None

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert result.stdout.endswith("Not redacted.\n\nNo speech detected.\n")
    assert "Language:" not in result.stdout
    assert fakes.calls == ["preflight", "decode", "language detection"]


def test_too_short_audio_skips_engines(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.seconds = 0.4

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert result.stdout.endswith("No speech detected.\n")
    assert "Language:" not in result.stdout
    assert fakes.calls == ["preflight", "decode"]


def test_too_short_audio_keeps_forced_language(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.seconds = 0.4

    result = run_main(str(audio_file), "--language", "de")

    assert result.exit_code == 0
    assert "Language: German (de), forced" in result.stdout
    assert fakes.calls == ["preflight", "decode"]


def test_no_turns_gives_unknown(audio_file: Path, fakes: FakeEngines) -> None:
    fakes.turns = []

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert "[00:00:00] UNKNOWN: Confidential words here." in result.stdout
    assert "SPEAKER_" not in result.stdout


@pytest.mark.parametrize("stage", ["decode", "language detection", "asr", "diarization"])
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


def test_default_transcribes_with_the_large_model(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert fakes.models_used == ["faster-whisper-large-v3-turbo"]


@pytest.mark.parametrize(("code", "model"), [("en", "faster-whisper-small.en"), ("it", "faster-whisper-small")])
def test_quick_transcribes_with_small_models(audio_file: Path, fakes: FakeEngines, code: str, model: str) -> None:
    fakes.language = Language(code, 0.9)

    result = run_main(str(audio_file), "--quick")

    assert result.exit_code == 0
    assert fakes.models_used == [model]


def test_host_and_profile_are_named_on_stderr(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert "transcribe: host Intel(R) Core(TM) i5-1038NG7, 16 GB, profile intel-mac" in result.stderr


@pytest.fixture
def apple_m2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_host", lambda: Host("apple-silicon", "Apple M2", 16 * 2**30))
    gpu = {"asr_backend": "mlx", "mlx_whisper_model": "mlx-whisper-large-v3"}
    monkeypatch.setitem(hardware.PROFILE_OVERRIDES, "apple-silicon", gpu)


@pytest.mark.usefixtures("apple_m2")
def test_apple_silicon_transcribes_with_its_profile(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert "profile apple-silicon" in result.stderr
    assert fakes.models_used == ["mlx-whisper-large-v3"]


@pytest.mark.usefixtures("apple_m2")
def test_quick_on_apple_silicon_uses_the_small_models(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--quick")

    assert result.exit_code == 0
    assert fakes.models_used == ["faster-whisper-small.en"]


@pytest.mark.usefixtures("apple_m2")
def test_forced_language_keeps_the_profile(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--language", "it")

    assert result.exit_code == 0
    assert fakes.models_used == ["mlx-whisper-large-v3"]


def test_rosetta_hints_at_an_arm64_python(
    audio_file: Path, fakes: FakeEngines, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "detect_host", lambda: Host("intel-mac", "Apple M2", 16 * 2**30, translated=True))

    result = run_main(str(audio_file))

    assert result.exit_code == 0
    assert "profile intel-mac" in result.stderr
    assert "arm64 Python" in result.stderr


@pytest.mark.parametrize(("backend", "expected"), [("mlx", "asr_mlx"), ("ctranslate2", "asr")])
def test_default_transcriber_follows_the_backend(monkeypatch: pytest.MonkeyPatch, backend: str, expected: str) -> None:
    from transcribe import asr, asr_mlx

    used: list[str] = []

    def fake(name: str) -> Callable[..., list[Word]]:
        def transcribe(*_args: object, **_kwargs: object) -> list[Word]:
            used.append(name)
            return []

        return transcribe

    monkeypatch.setattr(asr_mlx, "transcribe", fake("asr_mlx"))
    monkeypatch.setattr(asr, "transcribe", fake("asr"))
    settings = Settings(asr_backend=backend, mlx_whisper_model="mlx-whisper-large-v3")

    application._default_transcriber(np.zeros(16, np.float32), settings, "en", progress=lambda _d, _t: None)

    assert used == [expected]
