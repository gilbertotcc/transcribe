from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.metadata import version
from typing import TYPE_CHECKING

import pytest
from support import run_main


if TYPE_CHECKING:
    from pathlib import Path

    from support import FakeEngines


def test_help_exits_zero() -> None:
    result = run_main("--help")

    assert result.exit_code == 0
    assert "Usage: transcribe [OPTIONS] AUDIOFILE" in result.stdout
    assert "-o, --output OUTPUT.md" in result.stdout
    assert "--language CODE" in result.stdout
    assert "--quick" in result.stdout
    assert "--no-progress" in result.stdout
    assert "--version" in result.stdout
    assert result.stderr == ""


def test_help_starts_with_examples_and_ends_with_links() -> None:
    help_text = run_main("--help").stdout

    examples = help_text.index("Examples:")
    assert examples < help_text.index("The spoken language")
    assert "transcribe meeting.m4a -o meeting.md" in help_text
    assert help_text.rstrip().endswith("Issues: https://github.com/gilbertotcc/transcribe/issues")
    assert "Docs:   https://github.com/gilbertotcc/transcribe#readme" in help_text


def test_short_help_matches_long_help() -> None:
    short = run_main("-h")

    assert short.exit_code == 0
    assert short.stdout == run_main("--help").stdout


def test_version_prints_package_version() -> None:
    result = run_main("--version")

    assert result.exit_code == 0
    assert result.stdout == f"transcribe {version('transcribe')}\n"
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


def test_short_output_flag_validates_like_output(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "-o", str(tmp_path / "out.txt"))

    assert result.exit_code == 2
    assert ".md extension" in result.stderr
    assert not (tmp_path / "out.txt").exists()


def test_short_output_flag_writes_file(audio_file: Path, tmp_path: Path, fakes: FakeEngines) -> None:
    result = run_main(str(audio_file), "-o", str(tmp_path / "out.md"))

    assert result.exit_code == 0
    assert result.stdout == ""
    assert (tmp_path / "out.md").is_file()


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


@pytest.mark.parametrize(("flag", "expected"), [("--help", "Usage: transcribe"), ("--version", "transcribe ")])
def test_python_m_info_flags_do_not_load_engines(flag: str, expected: str) -> None:
    child = textwrap.dedent(
        f"""
        import runpy, sys
        sys.argv = ["transcribe", "{flag}"]
        try:
            runpy.run_module("transcribe", run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (0, None), exc.code
        heavy = {{"faster_whisper", "ctranslate2", "sherpa_onnx", "onnxruntime", "av", "numpy", "mlx", "mlx_whisper"}}
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
    assert completed.stdout.startswith(expected)
    assert completed.stderr == ""
