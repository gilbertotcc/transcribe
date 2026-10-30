"""Host detection and the hardware profile that adapts `Settings` to it (no engine imports)."""

from __future__ import annotations

import dataclasses
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Literal

from transcribe.settings import Settings


HostKind = Literal["apple-silicon", "intel-mac", "other"]
PROFILES: tuple[HostKind, ...] = ("apple-silicon", "intel-mac", "other")
_SYSCTL = "/usr/sbin/sysctl"

# Settings that each profile changes; every other field keeps its default. The Intel profile is the
# benchmarked reference (docs/benchmark.md), so it changes nothing. Apple Silicon transcribes on the GPU:
# twice as fast at the same accuracy on the 60 min recording (docs/benchmark.md#apple-silicon).
PROFILE_OVERRIDES: dict[HostKind, dict[str, Any]] = {
    "apple-silicon": {"asr_backend": "mlx", "mlx_whisper_model": "mlx-whisper-large-v3-turbo"},
    "intel-mac": {},
    "other": {},
}


@dataclass(frozen=True, slots=True)
class Host:
    kind: HostKind
    chip: str  # CPU brand string, "unknown" when it cannot be read
    memory_bytes: int  # 0 when unknown
    translated: bool = False  # an x86_64 Python under Rosetta 2 on Apple Silicon

    def describe(self) -> str:
        memory = f", {self.memory_bytes / 2**30:.0f} GB" if self.memory_bytes else ""
        rosetta = ", Rosetta" if self.translated else ""
        return f"{self.chip}{memory}{rosetta}"


def sysctl(name: str) -> str:
    """One `sysctl -n` value, or "unknown" when it cannot be read."""
    try:
        return subprocess.run(  # noqa: S603 - fixed system binary
            [_SYSCTL, "-n", name], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def detect_host() -> Host:
    """Classify the machine; anything unexpected is `other`, which keeps the default settings."""
    if sys.platform != "darwin":
        return Host("other", platform.processor() or platform.machine() or "unknown", 0)
    chip = sysctl("machdep.cpu.brand_string")
    memory = sysctl("hw.memsize")
    memory_bytes = int(memory) if memory.isdigit() else 0
    machine = platform.machine()
    if machine == "arm64":
        return Host("apple-silicon", chip, memory_bytes)
    if machine == "x86_64":
        translated = sysctl("sysctl.proc_translated") == "1"
        return Host("intel-mac", chip, memory_bytes, translated=translated)
    return Host("other", chip, memory_bytes)


def profile_for(host: Host, settings: Settings) -> HostKind:
    """The profile to run with: Apple Silicon needs enough memory, else it uses the Intel profile."""
    if host.kind == "apple-silicon" and host.memory_bytes < settings.apple_silicon_min_memory_bytes:
        return "intel-mac"
    return host.kind


def apply_profile(profile: HostKind, settings: Settings) -> Settings:
    """`settings` with the profile's overrides; fields the caller already changed from the default win."""
    defaults = Settings()
    changes = {
        name: value
        for name, value in PROFILE_OVERRIDES[profile].items()
        if getattr(settings, name) == getattr(defaults, name)
    }
    return dataclasses.replace(settings, **changes)
