"""Command-line interface: `transcribe AUDIOFILE [--output OUTPUT.md] [--language CODE] [--quick] [--no-progress]`."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import click

from transcribe.config import Settings
from transcribe.config.hardware import apply_profile, detect_host, profile_for
from transcribe.errors import INTERRUPTED_EXIT_CODE, InputPathError, TranscribeError
from transcribe.languages import normalize_code
from transcribe.pipeline.steps.output import validate_output_path


def _language_code(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_code(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


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
@click.option(
    "--language",
    metavar="CODE",
    callback=_language_code,
    help="Spoken language as a Whisper code (for example it); skips detection.",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Transcribe with Whisper small instead of large-v3-turbo: faster, less accurate.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Hide the progress bar (or progress lines) of speech recognition and diarization on stderr.",
)
def cli(audio_file: Path, output_path: Path | None, language: str | None, *, quick: bool, no_progress: bool) -> None:
    """Transcribe one local audio file with anonymous speaker labels.

    The spoken language is detected unless --language is given; one language
    per file, in any language Whisper supports.

    Everything runs on this machine; no audio or text leaves it. The transcript
    is raw and not redacted.
    """
    if audio_file.stat().st_size == 0:
        msg = f"audio file is empty: {audio_file}"
        raise InputPathError(msg)
    if output_path is not None:
        validate_output_path(output_path)  # fail fast, before loading any model

    from transcribe.cli import progress
    from transcribe.cli.console import ConsoleObserver
    from transcribe.pipeline import run  # heavy imports stay off --help

    host = detect_host()
    profile = profile_for(host, Settings())
    click.echo(f"transcribe: host {host.describe()}, profile {profile}", err=True)
    if host.translated:
        click.echo("transcribe: hint: an arm64 Python can use the Apple Silicon profile", err=True)
    settings = apply_profile(profile, Settings(language=language))
    if quick:
        settings = dataclasses.replace(settings, whisper_model_large=None)
    run(audio_file, output_path, settings, observer=ConsoleObserver(progress.reporter(enabled=not no_progress)))


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
