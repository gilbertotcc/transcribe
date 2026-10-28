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
_WHISPER_FILES = ("config.json", "model.bin", "tokenizer.json")
_VOCABULARY_FILES = ("vocabulary.json", "vocabulary.txt")  # CTranslate2 conversions write one of them


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


def whisper_paths(settings: Settings, model: str) -> tuple[Path, ...]:
    """Files of one Whisper model: its manifest entry, or the usual CTranslate2 layout if it has none."""
    assets = [asset for asset in load_manifest() if asset.name == model]
    listed = tuple(settings.models_root / f.path for asset in assets for f in asset.files)
    if listed:
        return listed
    directory = settings.models_root / model
    vocabulary = next((name for name in _VOCABULARY_FILES if (directory / name).is_file()), _VOCABULARY_FILES[-1])
    return tuple(directory / name for name in (*_WHISPER_FILES, vocabulary))


def required_paths(settings: Settings) -> tuple[Path, ...]:
    """Files the pipeline opens, given the configured model choices."""
    whisper = tuple(path for model in settings.whisper_models for path in whisper_paths(settings, model))
    recluster = settings.recluster_embedding_model_path
    diarization = (settings.segmentation_model_path, settings.embedding_model_path)
    return (*whisper, *diarization, *(() if recluster is None else (recluster,)))


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
