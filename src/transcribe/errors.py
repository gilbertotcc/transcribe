"""Errors raised by transcribe, each carrying the process exit code it maps to."""

USAGE_EXIT_CODE = 2
FAILURE_EXIT_CODE = 1
INTERRUPTED_EXIT_CODE = 130


class TranscribeError(Exception):
    """Base error; the message must never contain recognized text."""

    exit_code = FAILURE_EXIT_CODE


class InputPathError(TranscribeError):
    """The audio file argument is unusable (for example empty)."""

    exit_code = USAGE_EXIT_CODE


class OutputPathError(TranscribeError):
    """The `--output` destination is refused before any inference."""

    exit_code = USAGE_EXIT_CODE


class MissingModelsError(TranscribeError):
    """Required model files are absent from the models root."""

    exit_code = USAGE_EXIT_CODE


class DecodeError(TranscribeError):
    """The audio could not be decoded."""


class InferenceError(TranscribeError):
    """A speech recognition or diarization engine failed."""


class AttributionError(TranscribeError):
    """Engine results could not be combined into a transcript."""


class OutputError(TranscribeError):
    """The transcript could not be written."""
