from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

from support import run_main


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from support import FakeEngines


def test_help_exits_zero() -> None:
    result = run_main("--help")

    assert result.exit_code == 0
    assert "Usage: transcribe [OPTIONS] AUDIOFILE" in result.stdout
    assert "--output OUTPUT.md" in result.stdout
    assert "--language CODE" in result.stdout
    assert "--quick" in result.stdout
    assert "--no-progress" in result.stdout
    assert result.stderr == ""


def test_missing_input_argument() -> None:
    result = run_main()

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Missing argument 'AUDIOFILE'" in result.stderr


def test_nonexistent_input(tmp_path: Path) -> None:
    result = run_main(str(tmp_path / "nope.wav"))

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "does not exist" in result.stderr


def test_extra_argument(audio_file: Path) -> None:
    result = run_main(str(audio_file), str(audio_file))

    assert result.exit_code == 2
    assert "unexpected extra argument" in result.stderr


def test_unknown_option(audio_file: Path) -> None:
    result = run_main(str(audio_file), "--model", "base.en")

    assert result.exit_code == 2
    assert "No such option '--model'" in result.stderr


def test_directory_input(tmp_path: Path) -> None:
    result = run_main(str(tmp_path))

    assert result.exit_code == 2
    assert "is a directory" in result.stderr


def test_empty_input(tmp_path: Path, fakes: FakeEngines) -> None:
    empty = tmp_path / "empty.wav"
    empty.touch()

    result = run_main(str(empty))

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "audio file is empty" in result.stderr
    assert fakes.calls == []


def test_double_dash_allows_dash_prefixed_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fakes: FakeEngines
) -> None:
    (tmp_path / "-meeting.wav").write_bytes(b"RIFF")
    monkeypatch.chdir(tmp_path)

    result = run_main("--", "-meeting.wav")

    assert result.exit_code == 0
    assert result.stdout.startswith("# Transcript: -meeting.wav\n")


def test_unicode_and_spaces_in_input_path(tmp_path: Path, fakes: FakeEngines) -> None:
    audio = tmp_path / "réunion 01.wav"
    audio.write_bytes(b"RIFF")

    result = run_main(str(audio))

    assert result.exit_code == 0
    assert result.stdout.startswith("# Transcript: réunion 01.wav\n")


def test_output_existing_file_refused(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    existing = tmp_path / "out.md"
    existing.write_text("keep me")

    result = run_main(str(audio_file), "--output", str(existing))

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "already exists" in result.stderr
    assert existing.read_text() == "keep me"
    assert fakes.calls == []  # refused before any inference


def test_output_dangling_symlink_refused(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    link = tmp_path / "out.md"
    link.symlink_to(tmp_path / "missing-target.md")

    result = run_main(str(audio_file), "--output", str(link))

    assert result.exit_code == 2
    assert "already exists" in result.stderr
    assert not (tmp_path / "missing-target.md").exists()


def test_output_wrong_extension_refused(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--output", str(tmp_path / "out.txt"))

    assert result.exit_code == 2
    assert ".md extension" in result.stderr
    assert not (tmp_path / "out.txt").exists()


def test_output_dash_refused(audio_file: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--output", "-")

    assert result.exit_code == 2
    assert ".md extension" in result.stderr


def test_output_missing_parent_refused(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--output", str(tmp_path / "missing" / "out.md"))

    assert result.exit_code == 2
    assert "directory does not exist" in result.stderr
    assert not (tmp_path / "missing").exists()


def test_output_uppercase_extension_accepted(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "--output", str(tmp_path / "OUT.MD"))

    assert result.exit_code == 0
    assert (tmp_path / "OUT.MD").is_file()


def test_python_m_help_does_not_load_engines() -> None:
    child = textwrap.dedent(
        """
        import runpy, sys
        sys.argv = ["transcribe", "--help"]
        try:
            runpy.run_module("transcribe", run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (0, None), exc.code
        heavy = {"faster_whisper", "ctranslate2", "sherpa_onnx", "onnxruntime", "av", "numpy", "mlx", "mlx_whisper"}
        sys.stderr.write(",".join(sorted(heavy & set(sys.modules))))
        """
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "PYTHONWARNINGS": "error"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "Usage: transcribe" in completed.stdout
    assert completed.stderr == ""
