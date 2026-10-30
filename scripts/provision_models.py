"""Developer tool: download the pinned model files and verify their sha256.

The installed CLI never downloads anything; run this once per machine:

    uv run scripts/provision_models.py            # download + verify
    uv run scripts/provision_models.py --record   # first run: write missing hashes into the manifest
    uv run scripts/provision_models.py --all      # include the MLX models on an Intel Mac too

MLX models are provisioned only on Apple Silicon hosts that use the GPU profile, unless --all or --only.
"""

from __future__ import annotations

import json
import shutil
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import truststore

from transcribe.config import DEFAULT_MODELS_ROOT, Settings
from transcribe.config.hardware import detect_host, profile_for
from transcribe.models import load_manifest, sha256_file


if TYPE_CHECKING:
    from transcribe.models import Asset, AssetFile

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "src" / "transcribe" / "models" / "model_manifest.json"
_CHUNK = 1024 * 1024
# Verify TLS against the OS trust store: python.org builds ship without a CA bundle.
_TLS = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class ProvisioningError(click.ClickException):
    pass


def download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        msg = f"refusing non-HTTPS URL: {url}"
        raise ProvisioningError(msg)
    click.echo(f"  downloading {url}", err=True)
    with urllib.request.urlopen(url, timeout=60, context=_TLS) as response, destination.open("wb") as out:  # noqa: S310 - https checked above
        shutil.copyfileobj(response, out, _CHUNK)


def check_hash(path: Path, expected: str | None, label: str, recorded: dict[str, str], *, record: bool) -> None:
    actual = sha256_file(path)
    if expected is None:
        if not record:
            msg = f"{label}: manifest has no sha256; rerun with --record on a trusted network"
            raise ProvisioningError(msg)
        recorded[label] = actual
        click.echo(f"  recorded sha256 {actual} for {label}", err=True)
    elif actual != expected:
        msg = f"{label}: sha256 mismatch (expected {expected}, got {actual})"
        raise ProvisioningError(msg)


def is_current(root: Path, file: AssetFile) -> bool:
    target = root / file.path
    return target.is_file() and file.sha256 is not None and sha256_file(target) == file.sha256


def provision(asset: Asset, root: Path, scratch: Path, recorded: dict[str, str], *, record: bool) -> None:
    pending = [f for f in asset.files if not is_current(root, f)]
    if not pending:
        click.echo(f"{asset.name}: up to date", err=True)
        return
    click.echo(f"{asset.name} ({asset.license}, {asset.revision})", err=True)
    staged: list[tuple[AssetFile, Path]] = []
    if asset.archive_url is not None:
        archive = scratch / f"{asset.name}.archive"
        download(asset.archive_url, archive)
        check_hash(archive, asset.archive_sha256, asset.archive_url, recorded, record=record)
        with tarfile.open(archive) as tar:
            for file in pending:
                assert file.member is not None  # noqa: S101 - manifest invariant
                source = tar.extractfile(tar.getmember(file.member))
                if source is None:
                    msg = f"{asset.name}: {file.member} is not a regular file in the archive"
                    raise ProvisioningError(msg)
                staged_path = scratch / f"{asset.name}-{len(staged)}"
                with source, staged_path.open("wb") as out:
                    shutil.copyfileobj(source, out, _CHUNK)
                staged.append((file, staged_path))
    else:
        for file in pending:
            assert file.url is not None  # noqa: S101 - manifest invariant
            staged_path = scratch / f"{asset.name}-{len(staged)}"
            download(file.url, staged_path)
            staged.append((file, staged_path))
    for file, staged_path in staged:
        check_hash(staged_path, file.sha256, file.path, recorded, record=record)
        target = root / file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        staged_path.replace(target)
        click.echo(f"  installed {file.path}", err=True)


def write_recorded_hashes(recorded: dict[str, str]) -> None:
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text("utf-8"))
    for entry in manifest["assets"]:
        archive = entry.get("archive")
        if archive and archive["url"] in recorded:
            archive["sha256"] = recorded[archive["url"]]
        for file in entry["files"]:
            if file["path"] in recorded:
                file["sha256"] = recorded[file["path"]]
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    click.echo(f"wrote {len(recorded)} hashes to {MANIFEST_PATH}", err=True)


def is_mlx(asset: Asset) -> bool:
    """MLX models run only on the Apple Silicon profile (transcribe.config.hardware)."""
    return asset.name.startswith("mlx-")


@click.command()
@click.option(
    "--models-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_MODELS_ROOT,
    show_default=True,
)
@click.option("--record", is_flag=True, help="Record hashes that are missing from the manifest.")
@click.option("--only", "only", multiple=True, help="Provision only the named asset (repeatable).")
@click.option("--all", "everything", is_flag=True, help="Also provision the MLX models on hosts that cannot use them.")
def main(models_root: Path, *, record: bool, only: tuple[str, ...], everything: bool) -> None:
    """Download and verify the model files listed in model_manifest.json."""
    gpu = everything or bool(only) or profile_for(detect_host(), Settings()) == "apple-silicon"
    assets = [a for a in load_manifest() if (not only or a.name in only) and (gpu or not is_mlx(a))]
    models_root.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, str] = {}
    # Scratch space inside the models root keeps the final rename on one filesystem.
    with tempfile.TemporaryDirectory(dir=models_root, prefix=".download-") as scratch:
        for asset in assets:
            provision(asset, models_root, Path(scratch), recorded, record=record)
    if recorded:
        write_recorded_hashes(recorded)
    click.echo(f"models root: {models_root}", err=True)
    for asset in assets:
        click.echo(f"  {asset.name}: {asset.license}, {asset.source}", err=True)


if __name__ == "__main__":
    main()
