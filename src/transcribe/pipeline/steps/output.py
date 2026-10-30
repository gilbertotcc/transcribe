"""Transcript publication: stdout, or a new file that is never overwritten."""

from __future__ import annotations

import errno
import io
import os
import sys
import tempfile
from pathlib import Path

from transcribe.errors import OutputError, OutputPathError


_LINK_UNSUPPORTED = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}


def validate_output_path(path: Path) -> None:
    """Refuse destinations we would not publish to, before any inference runs."""
    if path.suffix.lower() != ".md":
        msg = f"output file must have a .md extension: {path}"
        raise OutputPathError(msg)
    if os.path.lexists(path):  # also catches dangling symlinks
        msg = f"output file already exists, refusing to overwrite: {path}"
        raise OutputPathError(msg)
    if not path.parent.is_dir():
        msg = f"output directory does not exist: {path.parent}"
        raise OutputPathError(msg)


def write_stdout(document: str) -> None:
    """Write the whole document to stdout at once, as UTF-8."""
    try:
        sys.stdout.buffer.write(document.encode("utf-8"))
        sys.stdout.flush()
    except BrokenPipeError:
        _silence_stdout()
        msg = "stdout was closed before the transcript was written"
        raise OutputError(msg) from None


def _silence_stdout() -> None:
    """Point stdout at /dev/null so the interpreter's final flush cannot raise again."""
    try:
        fileno = sys.stdout.fileno()
    except (OSError, ValueError, io.UnsupportedOperation):
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fileno)
    finally:
        os.close(devnull)


def publish_file(document: str, destination: Path) -> None:
    """Write to a private temporary file, then hard-link it into place without clobbering."""
    try:
        fd, name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    except OSError as exc:
        msg = f"cannot create a file in {destination.parent}: {exc.strerror}"
        raise OutputError(msg) from None
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(document.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        destination.hardlink_to(temporary)  # unlike os.replace, fails if destination exists
    except FileExistsError:
        msg = f"output file appeared while transcribing, refusing to overwrite: {destination}"
        raise OutputError(msg) from None
    except OSError as exc:
        if exc.errno in _LINK_UNSUPPORTED:
            msg = f"the filesystem of {destination.parent} does not support hard links; choose another directory"
        else:
            msg = f"cannot write {destination}: {exc.strerror or type(exc).__name__}"
        raise OutputError(msg) from None
    finally:
        temporary.unlink(missing_ok=True)
