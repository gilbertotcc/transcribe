"""Host detection and hardware profiles (no engines, no real sysctl)."""

from __future__ import annotations

import dataclasses
import subprocess
from typing import TYPE_CHECKING

import pytest

from transcribe import hardware
from transcribe.hardware import Host, apply_profile, detect_host, profile_for
from transcribe.settings import Settings


if TYPE_CHECKING:
    from collections.abc import Mapping

GB = 2**30
M2_SYSCTL = {"machdep.cpu.brand_string": "Apple M2", "hw.memsize": str(16 * GB)}
INTEL_SYSCTL = {"machdep.cpu.brand_string": "Intel(R) Core(TM) i5-1038NG7", "hw.memsize": str(16 * GB)}


def fake_host(monkeypatch: pytest.MonkeyPatch, platform: str, machine: str, values: Mapping[str, str]) -> None:
    monkeypatch.setattr("transcribe.hardware.sys.platform", platform)
    monkeypatch.setattr("transcribe.hardware.platform.machine", lambda: machine)
    monkeypatch.setattr(hardware, "sysctl", lambda name: values.get(name, "unknown"))


def test_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, "darwin", "arm64", M2_SYSCTL)

    assert detect_host() == Host("apple-silicon", "Apple M2", 16 * GB)


def test_intel_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, "darwin", "x86_64", INTEL_SYSCTL)

    host = detect_host()

    assert host.kind == "intel-mac"
    assert not host.translated


def test_rosetta_is_an_intel_mac_that_knows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, "darwin", "x86_64", {**M2_SYSCTL, "sysctl.proc_translated": "1"})

    host = detect_host()

    assert host.kind == "intel-mac"
    assert host.translated
    assert "Rosetta" in host.describe()


def test_unreadable_sysctl_keeps_the_kind_with_unknown_details(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, "darwin", "arm64", {})

    assert detect_host() == Host("apple-silicon", "unknown", 0)


def test_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, "linux", "aarch64", M2_SYSCTL)

    assert detect_host().kind == "other"


def test_sysctl_failure_reads_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "sysctl")

    monkeypatch.setattr("transcribe.hardware.subprocess.run", fail)

    assert hardware.sysctl("hw.memsize") == "unknown"


def test_describe_names_chip_and_memory() -> None:
    assert Host("apple-silicon", "Apple M2", 16 * GB).describe() == "Apple M2, 16 GB"


@pytest.mark.parametrize(
    ("host", "profile"),
    [
        (Host("apple-silicon", "Apple M2", 16 * GB), "apple-silicon"),
        (Host("apple-silicon", "Apple M1", 8 * GB), "intel-mac"),  # too little memory for the GPU profile
        (Host("apple-silicon", "unknown", 0), "intel-mac"),
        (Host("intel-mac", "Intel", 16 * GB), "intel-mac"),
        (Host("intel-mac", "Apple M2", 16 * GB, translated=True), "intel-mac"),
        (Host("other", "unknown", 0), "other"),
    ],
)
def test_profile_for(host: Host, profile: str) -> None:
    assert profile_for(host, Settings()) == profile


def test_intel_profile_is_the_defaults() -> None:
    assert apply_profile("intel-mac", Settings()) == Settings()
    assert apply_profile("other", Settings()) == Settings()


def test_profile_overrides_defaults_but_not_explicit_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {"asr_backend": "mlx", "mlx_whisper_model": "mlx-whisper-large-v3", "language": "fr"}
    monkeypatch.setitem(hardware.PROFILE_OVERRIDES, "apple-silicon", overrides)

    settings = apply_profile("apple-silicon", Settings(language="it"))

    assert settings.asr_backend == "mlx"
    assert settings.mlx_whisper_model == "mlx-whisper-large-v3"
    assert settings.language == "it"  # changed by the caller, so the profile leaves it


def test_shipped_profiles_name_real_settings() -> None:
    names = {field.name for field in dataclasses.fields(Settings)}

    for overrides in hardware.PROFILE_OVERRIDES.values():
        assert set(overrides) <= names
        apply_profile("intel-mac", Settings(**overrides))  # the values have valid types for the dataclass


def test_mlx_model_selection() -> None:
    settings = Settings(asr_backend="mlx", mlx_whisper_model="mlx-whisper-large-v3")

    assert settings.uses_mlx
    assert settings.whisper_model_for("it") == "mlx-whisper-large-v3"
    assert settings.whisper_models == ("faster-whisper-small", "mlx-whisper-large-v3")


def test_quick_leaves_mlx() -> None:
    settings = Settings(asr_backend="mlx", mlx_whisper_model="mlx-whisper-large-v3", whisper_model_large=None)

    assert not settings.uses_mlx
    assert settings.whisper_model_for("en") == "faster-whisper-small.en"
