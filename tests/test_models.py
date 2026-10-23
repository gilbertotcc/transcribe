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
