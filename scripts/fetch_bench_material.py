"""Developer tool: download the public-domain benchmark sources and rebuild every clip and reference.

    uv run scripts/fetch_bench_material.py [--bench-dir bench] [--record]

`scripts/bench_material.json` is the recipe: source URLs with their sha256, the
clips cut from them, the cast of the play and the simulated conversations. It
holds no audio and no text. Everything this script writes goes to the
git-ignored `bench/` directory (see docs/benchmark.md, section 2). Existing
files are kept, so a rerun only builds what is missing; `--record` writes the
hashes of sources that the recipe lacks, as `provision_models.py --record` does.
"""

from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import truststore

from transcribe.audio import decode
from transcribe.models import sha256_file
from transcribe.settings import Settings


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

SCRIPTS = Path(__file__).resolve().parent
RECIPE_PATH = SCRIPTS / "bench_material.json"
SAMPLE_RATE = 16_000
_CHUNK = 1024 * 1024
_TLS = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # the OS trust store, as in provision_models.py


class MaterialError(click.ClickException):
    pass


@dataclass(frozen=True, slots=True)
class Source:
    path: str  # relative to the bench directory
    url: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class Clip:
    """`seconds` of the concatenated `sources`, from `skip` on (None: to the end), after `lead_silence`."""

    path: str
    sources: tuple[str, ...]
    skip: float
    seconds: float | None
    lead_silence: float = 0.0


@dataclass(frozen=True, slots=True)
class Reference:
    path: str
    start: str  # line where the reference starts (inclusive)
    end: str  # line where it stops (exclusive)


@dataclass(frozen=True, slots=True)
class Simulation:
    split: str
    name: str  # language tag in the output names
    minutes: float
    reference: bool  # True: ASR set (clips/asr, no cropping); False: diarization set (clips/sim)

    def stem(self, seed: int) -> str:
        return f"{'asr' if self.reference else 'sim'}_{self.name}_s{seed}"

    def folder(self) -> str:
        return "clips/asr" if self.reference else "clips/sim"


@dataclass(frozen=True, slots=True)
class Recipe:
    sources: tuple[Source, ...]
    clips: tuple[Clip, ...]
    play_text: str
    cast: dict[str, str]
    references: tuple[Reference, ...]
    simulations: tuple[Simulation, ...]
    seeds: tuple[int, ...]


def load_recipe(path: Path = RECIPE_PATH) -> Recipe:
    raw: dict[str, Any] = json.loads(path.read_text("utf-8"))
    clips = []
    for entry in raw["clips"]:
        if entry["skip"] is None:
            msg = f"{entry['path']}: the recipe has no start offset"
            raise MaterialError(msg)
        clips.append(
            Clip(
                path=entry["path"],
                sources=tuple(entry["sources"]),
                skip=float(entry["skip"]),
                seconds=None if entry["seconds"] is None else float(entry["seconds"]),
                lead_silence=float(entry.get("lead_silence", 0.0)),
            )
        )
    play = raw["play"]
    return Recipe(
        sources=tuple(Source(**entry) for entry in raw["sources"]),
        clips=tuple(clips),
        play_text=play["text"],
        cast=dict(play["cast"]),
        references=tuple(Reference(**entry) for entry in play["references"]),
        simulations=tuple(Simulation(**entry) for entry in raw["simulations"]),
        seeds=tuple(raw["seeds"]),
    )


# --- Sources -----------------------------------------------------------------


def download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        msg = f"refusing non-HTTPS URL: {url}"
        raise MaterialError(msg)
    click.echo(f"  downloading {url}", err=True)
    with urllib.request.urlopen(url, timeout=60, context=_TLS) as response, destination.open("wb") as out:  # noqa: S310 - https checked above
        shutil.copyfileobj(response, out, _CHUNK)


def fetch_sources(recipe: Recipe, bench: Path, *, record: bool) -> dict[str, str]:
    """Download missing sources and verify every hash; returns the hashes recorded with `record`."""
    recorded: dict[str, str] = {}
    for source in recipe.sources:
        target = bench / source.path
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent, prefix=".download-") as scratch:
                staged = Path(scratch) / target.name
                download(source.url, staged)
                staged.rename(target)
        actual = sha256_file(target)
        if source.sha256 is None:
            if not record:
                msg = f"{source.path}: the recipe has no sha256; rerun with --record on a trusted network"
                raise MaterialError(msg)
            recorded[source.path] = actual
        elif actual != source.sha256:
            msg = f"{source.path}: sha256 mismatch (expected {source.sha256}, got {actual})"
            raise MaterialError(msg)
    return recorded


def write_recorded_hashes(recorded: dict[str, str], path: Path = RECIPE_PATH) -> None:
    raw: dict[str, Any] = json.loads(path.read_text("utf-8"))
    for entry in raw["sources"]:
        if entry["path"] in recorded:
            entry["sha256"] = recorded[entry["path"]]
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", "utf-8")
    click.echo(f"wrote {len(recorded)} hashes to {path}", err=True)


# --- Clips -------------------------------------------------------------------


def cut(
    parts: Sequence[npt.NDArray[np.float32]], skip: float, seconds: float | None, lead_silence: float = 0.0
) -> npt.NDArray[np.float32]:
    """Concatenate `parts`, keep `seconds` from `skip` on (None: to the end), prepend `lead_silence` of zeros."""
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    begin = round(skip * SAMPLE_RATE)
    if begin >= len(audio):
        msg = f"the sources last {len(audio) / SAMPLE_RATE:.1f} s, less than the {skip} s to skip"
        raise MaterialError(msg)
    end = len(audio) if seconds is None else begin + round(seconds * SAMPLE_RATE)
    if end > len(audio):
        msg = f"the sources last {len(audio) / SAMPLE_RATE:.1f} s, too short for {seconds} s from {skip} s"
        raise MaterialError(msg)
    silence = np.zeros(round(lead_silence * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([silence, audio[begin:end]])


def write_wav(path: Path, samples: npt.NDArray[np.float32]) -> None:
    with path.open("xb") as raw, wave.open(raw, "wb") as stream:  # "x": never replace an existing clip
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def write_m4a(path: Path, samples: npt.NDArray[np.float32]) -> None:
    import av  # PyAV, a dependency of faster-whisper

    with path.open("xb") as raw, av.open(raw, mode="w", format="ipod") as container:
        stream = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")
        stream.bit_rate = 64_000
        frame_size = 1024
        for begin in range(0, len(samples), frame_size):
            block = np.clip(samples[begin : begin + frame_size], -1.0, 1.0).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(block.astype(np.float32), format="fltp", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = begin
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def build_clips(recipe: Recipe, bench: Path) -> None:
    settings = Settings(sample_rate=SAMPLE_RATE)
    for clip in recipe.clips:
        target = bench / clip.path
        if target.exists():
            continue
        click.echo(f"building {clip.path}", err=True)
        parts = [decode(bench / source, settings) for source in clip.sources]
        samples = cut(parts, clip.skip, clip.seconds, clip.lead_silence)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            (write_m4a if target.suffix == ".m4a" else write_wav)(target, samples)
        except BaseException:
            target.unlink(missing_ok=True)
            raise


# --- References and simulations ----------------------------------------------


def run_script(name: str, args: Sequence[str]) -> str:
    """Run a sibling developer script with this interpreter and return its stdout."""
    command = [sys.executable, str(SCRIPTS / f"{name}.py"), *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603 - fixed scripts


def build_references(recipe: Recipe, bench: Path) -> None:
    cast = [f"--cast={character}={reader}" for character, reader in recipe.cast.items()]
    for reference in recipe.references:
        target = bench / reference.path
        if target.exists():
            continue
        click.echo(f"building {reference.path}", err=True)
        text = run_script(
            "play_reference",
            [str(bench / recipe.play_text), f"--start={reference.start}", f"--end={reference.end}", *cast],
        )
        with target.open("x", encoding="utf-8") as stream:
            stream.write(text)


def build_simulations(recipe: Recipe, bench: Path) -> None:
    for simulation in recipe.simulations:
        folder = bench / simulation.folder()
        folder.mkdir(parents=True, exist_ok=True)
        for seed in recipe.seeds:
            stem = simulation.stem(seed)
            if (folder / f"{stem}.wav").exists():
                continue
            click.echo(f"building {simulation.folder()}/{stem}.wav", err=True)
            args = [
                str(bench / simulation.split),
                "--speakers=6",
                f"--minutes={simulation.minutes}",
                f"--seed={seed}",
                f"--out={folder / f'{stem}.wav'}",
                f"--rttm={folder / f'{stem}.rttm'}",
            ]
            if simulation.reference:
                args.append(f"--reference={folder / f'{stem}.tsv'}")
            run_script("simulate_conversation", args)


@click.command()
@click.option(
    "--bench-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=SCRIPTS.parent / "bench",
    show_default=True,
)
@click.option("--record", is_flag=True, help="Record source hashes that are missing from the recipe.")
def main(bench_dir: Path, *, record: bool) -> None:
    """Download the benchmark sources, verify them and build the clips, references and simulations."""
    recipe = load_recipe()
    recorded = fetch_sources(recipe, bench_dir, record=record)
    if recorded:
        write_recorded_hashes(recorded)
    build_clips(recipe, bench_dir)
    build_references(recipe, bench_dir)
    build_simulations(recipe, bench_dir)
    click.echo(f"bench material ready in {bench_dir}", err=True)


if __name__ == "__main__":
    main()
