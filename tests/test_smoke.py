from typing import TYPE_CHECKING

from transcribe import main


if TYPE_CHECKING:
    import pytest


def test_main_greets(capsys: pytest.CaptureFixture[str]) -> None:
    main()

    assert capsys.readouterr().out == "Hello from transcribe!\n"
