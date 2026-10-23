from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

import pytest

from transcribe import output
from transcribe.errors import OutputError, OutputPathError


if TYPE_CHECKING:
    from pathlib import Path


def test_publish_creates_private_utf8_file(tmp_path: Path) -> None:
    destination = tmp_path / "out.md"

    output.publish_file("# Café\n", destination)

    assert destination.read_bytes() == "# Café\n".encode()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert destination.stat().st_nlink == 1  # temporary link removed
    assert [p.name for p in tmp_path.iterdir()] == ["out.md"]


def test_publish_refuses_file_that_appeared(tmp_path: Path) -> None:
    destination = tmp_path / "out.md"
    destination.write_text("someone else's")

    with pytest.raises(OutputError, match="refusing to overwrite"):
        output.publish_file("new", destination)

    assert destination.read_text() == "someone else's"
    assert [p.name for p in tmp_path.iterdir()] == ["out.md"]


def test_publish_removes_temporary_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_hard_links(self: Path, target: Path) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(type(tmp_path), "hardlink_to", no_hard_links)

    with pytest.raises(OutputError, match="does not support hard links"):
        output.publish_file("text", tmp_path / "out.md")

    assert list(tmp_path.iterdir()) == []


def test_publish_into_missing_directory_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(OutputError, match="cannot create a file"):
        output.publish_file("text", tmp_path / "gone" / "out.md")


@pytest.mark.parametrize("name", ["out.txt", "out", "-", "out.md.bak"])
def test_validate_rejects_non_markdown(tmp_path: Path, name: str) -> None:
    with pytest.raises(OutputPathError, match=r"\.md extension"):
        output.validate_output_path(tmp_path / name)


def test_validate_accepts_new_markdown_file(tmp_path: Path) -> None:
    output.validate_output_path(tmp_path / "new.Md")


class ClosedPipe(io.RawIOBase):
    def writable(self) -> bool:
        return True

    def write(self, b: object) -> int:
        raise BrokenPipeError


def test_write_stdout_broken_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BufferedWriter(ClosedPipe())))

    with pytest.raises(OutputError, match="stdout was closed"):
        output.write_stdout("x" * 100_000)


def test_write_stdout_is_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="ascii"))

    output.write_stdout("naïve\n")

    assert buffer.getvalue() == "naïve\n".encode()
