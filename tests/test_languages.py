from __future__ import annotations

import pytest

from transcribe.domain import Language
from transcribe.languages import LANGUAGES, label, normalize_code


@pytest.mark.parametrize(("raw", "code"), [("it", "it"), ("IT", "it"), (" en ", "en"), ("Yue", "yue")])
def test_normalize_accepts_known_codes_in_any_case(raw: str, code: str) -> None:
    assert normalize_code(raw) == code


@pytest.mark.parametrize("raw", ["xx", "", "ita", "italian", "en-US"])
def test_normalize_rejects_unknown_codes(raw: str) -> None:
    with pytest.raises(ValueError, match="unknown language code"):
        normalize_code(raw)


def test_label_names_the_language_and_code() -> None:
    assert label("it") == "Italian (it)"


def test_table_has_every_whisper_language() -> None:
    assert len(LANGUAGES) == 100
    assert all(code == code.lower() for code in LANGUAGES)


def test_forced_language_has_no_probability() -> None:
    assert Language("it", None).forced
    assert not Language("it", 0.9).forced
