"""Command-line interface: `transcribe AUDIOFILE [--output OUTPUT.md]`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from transcribe.errors import INTERRUPTED_EXIT_CODE, InputPathError, TranscribeError
from transcribe.output import validate_output_path
from transcribe.settings import Settings


@click.command(name="transcribe")
@click.argument(
    "audio_file",
    metavar="AUDIOFILE",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    metavar="OUTPUT.md",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write Markdown to a new .md file instead of stdout.",
)
def cli(audio_file: Path, output_path: Path | None) -> None:
    """Transcribe one local English audio file with anonymous speaker labels.

    Everything runs on this machine; no audio or text leaves it. The transcript
    is raw and not redacted.
    """
    if audio_file.stat().st_size == 0:
        msg = f"audio file is empty: {audio_file}"
        raise InputPathError(msg)
    if output_path is not None:
        validate_output_path(output_path)  # fail fast, before loading any model

    from transcribe.application import run  # heavy imports stay off --help

    run(audio_file, output_path, Settings())


def main() -> None:
    """Console-script entry point that maps outcomes to the documented exit codes."""
    try:
        code = cli.main(standalone_mode=False)
    except click.Abort:  # Click turns KeyboardInterrupt into Abort
        click.echo("transcribe: interrupted", err=True)
        sys.exit(INTERRUPTED_EXIT_CODE)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except TranscribeError as exc:
        click.echo(f"transcribe: {exc}", err=True)
        sys.exit(exc.exit_code)
    sys.exit(code if isinstance(code, int) else 0)
