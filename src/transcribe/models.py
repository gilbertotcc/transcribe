"""Model asset manifest and the runtime preflight check."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

from transcribe.errors import MissingModelsError


if TYPE_CHECKING:
    from pathlib import Path

    from transcribe.settings import Settings

PROVISIONING_HINT = "run `uv run scripts/provision_models.py` to download them"
_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class AssetFile:
    path: str  # relative to the models root
    sha256: str | None
    url: str | None = None  # direct download
    member: str | None = None  # path inside the asset's archive


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    license: str
    source: str
    revision: str
    files: tuple[AssetFile, ...]
    archive_url: str | None = None
    archive_sha256: str | None = None


def load_manifest() -> tuple[Asset, ...]:
    raw: dict[str, Any] = json.loads(resources.files("transcribe").joinpath("model_manifest.json").read_text("utf-8"))
    assets = []
    for entry in raw["assets"]:
        archive = entry.get("archive") or {}
        assets.append(
            Asset(
                name=entry["name"],
                license=entry["license"],
                source=entry["source"],
                revision=entry["revision"],
                files=tuple(AssetFile(**f) for f in entry["files"]),
                archive_url=archive.get("url"),
                archive_sha256=archive.get("sha256"),
            )
        )
    return tuple(assets)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def required_paths(settings: Settings) -> tuple[Path, ...]:
    """Files the pipeline opens, given the configured model choices."""
    whisper = settings.whisper_model_dir
    return (
        whisper / "config.json",
        whisper / "model.bin",
        whisper / "tokenizer.json",
        whisper / "vocabulary.txt",
        settings.segmentation_model_path,
        settings.embedding_model_path,
    )


def check_assets(settings: Settings) -> None:
    """Fail fast (exit 2) if a model file is missing or, optionally, has the wrong hash."""
    missing = [path for path in required_paths(settings) if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(settings.models_root)) for path in missing)
        msg = f"missing model files in {settings.models_root}: {names}; {PROVISIONING_HINT}"
        raise MissingModelsError(msg)
    if settings.verify_model_hashes:
        expected = {
            settings.models_root / f.path: f.sha256 for asset in load_manifest() for f in asset.files if f.sha256
        }
        for path in required_paths(settings):
            want = expected.get(path)
            if want is not None and sha256_file(path) != want:
                msg = f"model file has an unexpected sha256: {path.name}; {PROVISIONING_HINT}"
                raise MissingModelsError(msg)
