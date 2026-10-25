from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from transcribe.errors import MissingModelsError
from transcribe.models import check_assets, load_manifest, required_paths, whisper_paths
from transcribe.settings import Settings


if TYPE_CHECKING:
    from pathlib import Path


def test_manifest_pins_every_required_file() -> None:
    settings = Settings()
    manifest = {settings.models_root / f.path: f for asset in load_manifest() for f in asset.files}

    for path in required_paths(settings):
        assert path in manifest
        assert manifest[path].sha256 is not None


def test_manifest_records_licenses_and_revisions() -> None:
    for asset in load_manifest():
        assert asset.license
        assert asset.revision
        assert asset.source.startswith("https://")


def test_missing_models_name_the_provisioning_script(tmp_path: Path) -> None:
    with pytest.raises(MissingModelsError, match=r"provision_models\.py") as excinfo:
        check_assets(Settings(models_root=tmp_path))

    assert excinfo.value.exit_code == 2


def test_present_models_pass(tmp_path: Path) -> None:
    settings = Settings(models_root=tmp_path)
    for path in required_paths(settings):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")

    check_assets(settings)


def test_hash_mismatch_is_reported_when_verifying(tmp_path: Path) -> None:
    settings = Settings(models_root=tmp_path, verify_model_hashes=True)
    for path in required_paths(settings):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")

    with pytest.raises(MissingModelsError, match="unexpected sha256"):
        check_assets(settings)


def model_names(settings: Settings) -> set[str]:
    return {path.relative_to(settings.models_root).parts[0] for path in required_paths(settings)}


def test_default_requires_the_detection_and_large_models() -> None:
    names = model_names(Settings())

    assert {"faster-whisper-small", "faster-whisper-large-v3-turbo"} <= names
    assert "faster-whisper-small.en" not in names


def test_quick_requires_the_small_models_only() -> None:
    names = model_names(Settings(whisper_model_large=None))

    assert {"faster-whisper-small", "faster-whisper-small.en"} <= names
    assert "faster-whisper-large-v3-turbo" not in names


def test_english_model_can_be_left_out() -> None:
    settings = Settings(whisper_model_large=None, whisper_model_en=None)

    assert "faster-whisper-small.en" not in model_names(settings)
    assert settings.whisper_model_dir_for("en") == settings.whisper_model_dir


def test_routing_by_language() -> None:
    large, quick = Settings(), Settings(whisper_model_large=None)

    assert large.whisper_model_dir_for("en").name == "faster-whisper-large-v3-turbo"
    assert large.whisper_model_dir_for("it").name == "faster-whisper-large-v3-turbo"
    assert quick.whisper_model_dir_for("en").name == "faster-whisper-small.en"
    assert quick.whisper_model_dir_for("it").name == "faster-whisper-small"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"language": "it"}, {"faster-whisper-large-v3-turbo"}),
        ({"language": "en", "whisper_model_large": None}, {"faster-whisper-small.en"}),
        ({"language": "it", "whisper_model_large": None}, {"faster-whisper-small"}),
    ],
)
def test_forced_language_needs_no_detection_model(overrides: dict[str, Any], expected: set[str]) -> None:
    whisper = {name for name in model_names(Settings(**overrides)) if name.startswith("faster-whisper")}

    assert whisper == expected


@pytest.mark.parametrize("vocabulary", ["vocabulary.json", "vocabulary.txt"])
def test_unlisted_model_accepts_either_vocabulary_file(tmp_path: Path, vocabulary: str) -> None:
    settings = Settings(models_root=tmp_path)
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / vocabulary).write_text("{}", encoding="utf-8")

    assert [path.name for path in whisper_paths(settings, "custom")] == [
        "config.json",
        "model.bin",
        "tokenizer.json",
        vocabulary,
    ]
