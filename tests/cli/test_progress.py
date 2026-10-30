from __future__ import annotations

import io
import sys

import pytest

from transcribe.cli import progress


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def _lines(reporter: progress.Reporter, steps: list[tuple[float, float]], capsys: pytest.CaptureFixture[str]) -> str:
    with reporter.track("asr") as update:
        for done, total in steps:
            update(done, total)
    return capsys.readouterr().err


def test_line_reporter_prints_each_decile_once(capsys: pytest.CaptureFixture[str]) -> None:
    err = _lines(progress.LineReporter(), [(5, 60), (6, 60), (31, 60), (60, 60), (60, 60)], capsys)

    assert err.splitlines() == [
        "transcribe: asr 10%",
        "transcribe: asr 50%",
        "transcribe: asr 100%",
    ]


def test_line_reporter_ignores_unknown_total(capsys: pytest.CaptureFixture[str]) -> None:
    assert _lines(progress.LineReporter(), [(3, 0)], capsys) == ""


def test_null_reporter_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    assert _lines(progress.NullReporter(), [(30, 60), (60, 60)], capsys) == ""


def test_bar_reporter_draws_percentage_on_a_terminal() -> None:
    terminal = Terminal()

    with progress.BarReporter(terminal).track("diarization") as update:
        update(1, 4)
        update(4, 4)

    drawn = terminal.getvalue()
    assert "transcribe: diarization" in drawn
    assert "25%" in drawn
    assert "100%" in drawn
    assert drawn.endswith("\n")


def test_bar_reporter_ends_its_line_when_interrupted() -> None:
    terminal = Terminal()

    def interrupted_stage() -> None:
        with progress.BarReporter(terminal).track("asr") as update:
            update(1, 2)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted_stage()

    assert terminal.getvalue().endswith("\n")


@pytest.mark.parametrize(
    ("enabled", "tty", "expected"),
    [
        (True, True, progress.BarReporter),
        (True, False, progress.LineReporter),
        (False, True, progress.NullReporter),
        (False, False, progress.NullReporter),
    ],
)
def test_reporter_selection(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, tty: bool, expected: type) -> None:
    stream = Terminal() if tty else io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)

    assert isinstance(progress.reporter(enabled=enabled), expected)
