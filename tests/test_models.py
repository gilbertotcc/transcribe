from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from transcribe.errors import MissingModelsError
from transcribe.models import check_assets, load_manifest, required_paths
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


def test_both_whisper_models_are_required_by_default() -> None:
    settings = Settings()
    names = {path.relative_to(settings.models_root).parts[0] for path in required_paths(settings)}

    assert {"faster-whisper-small", "faster-whisper-small.en"} <= names


def test_english_model_can_be_left_out() -> None:
    settings = Settings(whisper_model_en=None)
    names = {path.relative_to(settings.models_root).parts[0] for path in required_paths(settings)}

    assert "faster-whisper-small.en" not in names
    assert settings.whisper_model_dir_for("en") == settings.whisper_model_dir


def test_english_routes_to_the_english_model() -> None:
    settings = Settings()

    assert settings.whisper_model_dir_for("en").name == "faster-whisper-small.en"
    assert settings.whisper_model_dir_for("it").name == "faster-whisper-small"
