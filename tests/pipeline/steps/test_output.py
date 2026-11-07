from __future__ import annotations

import errno
import io
import os
import sys
from pathlib import Path

import pytest

from transcribe.errors import OutputError, OutputPathError
from transcribe.pipeline.steps import output


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


def fail_hard_links(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    def hardlink_to(self: Path, target: Path) -> None:
        raise error

    monkeypatch.setattr(type(Path()), "hardlink_to", hardlink_to)


NO_HARD_LINKS = "does not support hard links"


@pytest.mark.parametrize(
    ("code", "expected", "forbidden"),
    [
        (errno.EPERM, ["may not support hard links", "may forbid them", "check its permissions"], [NO_HARD_LINKS]),
        (errno.ENOTSUP, [NO_HARD_LINKS, "choose another directory"], ["forbid"]),
        (errno.EOPNOTSUPP, [NO_HARD_LINKS], ["forbid"]),
        (errno.EMLINK, [NO_HARD_LINKS], ["forbid"]),
        (errno.EXDEV, ["spans more than one mount", "choose another directory"], [NO_HARD_LINKS]),
        (errno.EIO, ["cannot write", os.strerror(errno.EIO)], [NO_HARD_LINKS, "forbid"]),
    ],
)
def test_publish_explains_hard_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, expected: list[str], forbidden: list[str]
) -> None:
    fail_hard_links(monkeypatch, OSError(code, os.strerror(code)))

    with pytest.raises(OutputError) as caught:
        output.publish_file("text", tmp_path / "out.md")

    message = str(caught.value)
    for fragment in expected:
        assert fragment in message
    for fragment in forbidden:
        assert fragment not in message
    if code != errno.EIO:
        assert f"({errno.errorcode[code]})" in message
    assert caught.value.exit_code == 1
    assert list(tmp_path.iterdir()) == []  # temporary file removed


def test_publish_eperm_from_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail_hard_links(monkeypatch, PermissionError(errno.EPERM, "Operation not permitted"))

    with pytest.raises(OutputError, match="may not support hard links, or the system may forbid them"):
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
