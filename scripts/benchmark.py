"""Developer tool: run the pipeline on one recording and write a JSON report.

    uv run scripts/benchmark.py AUDIO --label NAME [--reference REF] [--set field=value ...]

The report holds measurements only: no recognized text and no file paths. Run
each measurement in a fresh process (peak RSS is per process).

REF is plain text, or `SPEAKER<TAB>text` lines (see scripts/play_reference.py)
to also score speaker attribution; `-` marks text without a known speaker.
The reference may cover more than the recording: both sides are trimmed to the
span where they align, and the trimmed word counts are reported.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import jiwer

from transcribe.application import build_transcript
from transcribe.models import load_manifest, required_paths
from transcribe.settings import DEFAULT_MODELS_ROOT, Settings


if TYPE_CHECKING:
    from collections.abc import Sequence

    from transcribe.domain import AttributedWord

GATES = {"rtf_max": 1.0, "peak_rss_max_bytes": 4_500_000_000, "wer_max": 0.08, "speaker_accuracy_min": 0.90}
UNSCORED = "-"
_DASHES = re.compile("[\u2014\u2013\\-/]")
_NOT_WORD = re.compile(r"[^a-z0-9' ]")
_MIN_ANCHOR = 4  # words in a matching block that anchors the alignment window
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # a plain file-name stem, so reports stay in --output-dir


# --- Host and settings -------------------------------------------------------


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed system binary
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def host_info() -> dict[str, Any]:
    packages = ["faster-whisper", "ctranslate2", "sherpa-onnx", "onnxruntime", "av", "numpy", "click"]
    return {
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "physical_cores": _sysctl("hw.physicalcpu"),
        "logical_cores": _sysctl("hw.logicalcpu"),
        "ram_bytes": _sysctl("hw.memsize"),
        "macos": platform.mac_ver()[0] or platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in packages},
    }


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def model_hashes(settings: Settings) -> dict[str, str | None]:
    manifest = {settings.models_root / f.path: f.sha256 for asset in load_manifest() for f in asset.files}
    return {str(path.relative_to(settings.models_root)): manifest.get(path) for path in required_paths(settings)}


def apply_overrides(items: Sequence[str]) -> Settings:
    fields = {field.name: field for field in dataclasses.fields(Settings)}
    changes: dict[str, Any] = {}
    for item in items:
        name, _, raw = item.partition("=")
        if name not in fields:
            msg = f"unknown setting {name!r}; choose from {', '.join(fields)}"
            raise click.BadParameter(msg)
        kind = fields[name].type
        if kind is bool:
            changes[name] = raw.lower() in {"1", "true", "yes"}
        elif kind is Path:
            changes[name] = Path(raw).expanduser()
        else:
            changes[name] = kind(raw)  # type: ignore[operator]
    return Settings(**changes)


def settings_report(settings: Settings) -> dict[str, Any]:
    values = dataclasses.asdict(settings)
    values["models_root"] = "default" if settings.models_root == DEFAULT_MODELS_ROOT else "custom"
    return values


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # macOS: bytes; Linux: KiB


# --- Reference scoring -------------------------------------------------------


def tokens(text: str) -> list[str]:
    text = _DASHES.sub(" ", text.lower().replace("\u2019", "'").replace("\u2018", "'"))
    return [token.strip("'") for token in _NOT_WORD.sub(" ", text).split() if token.strip("'")]


def read_reference(path: Path) -> tuple[list[str], list[str]]:
    words: list[str] = []
    speakers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        speaker, tab, text = line.partition("\t")
        if not tab:
            speaker, text = UNSCORED, line
        line_tokens = tokens(text)
        words.extend(line_tokens)
        speakers.extend([speaker] * len(line_tokens))
    return words, speakers


def hypothesis(words: Sequence[AttributedWord]) -> tuple[list[str], list[int | None]]:
    texts: list[str] = []
    speakers: list[int | None] = []
    for item in words:
        word_tokens = tokens(item.word.text)
        texts.extend(word_tokens)
        speakers.extend([item.speaker] * len(word_tokens))
    return texts, speakers


def aligned_window(ref: list[str], hyp: list[str]) -> tuple[slice, slice]:
    """Span of ref and hyp between the first and last well-anchored matching blocks."""
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= _MIN_ANCHOR]
    if not blocks:
        return slice(0, len(ref)), slice(0, len(hyp))
    first, last = blocks[0], blocks[-1]
    # Extend to the hypothesis edges, so unmatched words at the edges still count.
    ref_start = max(first.a - first.b, 0)
    ref_stop = min(last.a + last.size + (len(hyp) - last.b - last.size), len(ref))
    return slice(ref_start, ref_stop), slice(0, len(hyp))


def best_one_to_one(counts: dict[tuple[int, str], int], clusters: list[int], readers: list[str]) -> int:
    """Exact maximum of correctly attributed words over one-to-one cluster->reader maps."""
    reader_bit = {reader: 1 << index for index, reader in enumerate(readers)}
    best: dict[int, int] = {0: 0}  # used-readers mask -> best score
    for cluster in clusters:
        nxt = dict(best)
        for mask, score in best.items():
            for reader, bit in reader_bit.items():
                if not mask & bit:
                    candidate = score + counts.get((cluster, reader), 0)
                    if candidate > nxt.get(mask | bit, -1):
                        nxt[mask | bit] = candidate
        best = nxt
    return max(best.values())


def score(reference: Path, words: Sequence[AttributedWord], alignment_out: Path | None) -> dict[str, Any]:
    ref_all, ref_speakers_all = read_reference(reference)
    hyp, hyp_speakers = hypothesis(words)
    ref_window, _ = aligned_window(ref_all, hyp)
    ref, ref_speakers = ref_all[ref_window], ref_speakers_all[ref_window]
    output = jiwer.process_words(" ".join(ref), " ".join(hyp))
    if alignment_out is not None:  # contains recognized text: for hand-checking only
        alignment_out.write_text(jiwer.visualize_alignment(output, show_measures=False), encoding="utf-8")
    report: dict[str, Any] = {
        "wer": {
            "wer": round(output.wer, 4),
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "hits": output.hits,
            "reference_words": len(ref),
            "reference_words_before_trim": len(ref_all),
            "hypothesis_words": len(hyp),
        }
    }
    pairs: list[tuple[int | None, str]] = []
    for chunk in output.alignments[0]:
        if chunk.type in {"equal", "substitute"}:
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                reader = ref_speakers[chunk.ref_start_idx + offset]
                if reader != UNSCORED:
                    pairs.append((hyp_speakers[chunk.hyp_start_idx + offset], reader))
    if pairs:
        counts: dict[tuple[int, str], int] = {}
        for cluster, reader in pairs:
            if cluster is not None:
                counts[cluster, reader] = counts.get((cluster, reader), 0) + 1
        clusters = sorted({cluster for cluster, _ in counts})
        readers = sorted({reader for _, reader in pairs})
        majority = sum(max(counts.get((c, r), 0) for r in readers) for c in clusters)
        report["speaker_accuracy"] = {
            "one_to_one": round(best_one_to_one(counts, clusters, readers) / len(pairs), 4),
            "many_to_one": round(majority / len(pairs), 4),
            "scored_words": len(pairs),
            "unknown_words": sum(1 for cluster, _ in pairs if cluster is None),
            "reference_speakers": len(readers),
            "clusters_on_scored_words": len(clusters),
        }
    return report


def gates(report: dict[str, Any]) -> dict[str, str]:
    def status(value: float | None, ok: bool) -> str:  # noqa: FBT001 - local helper
        return "unknown" if value is None else ("passed" if ok else "failed")

    rtf = report["real_time_factor"]
    rss = report["peak_rss_bytes"]
    wer = report.get("wer", {}).get("wer")
    accuracy = report.get("speaker_accuracy", {}).get("one_to_one")
    return {
        "runtime": status(rtf, rtf is not None and rtf <= GATES["rtf_max"]),
        "memory": status(rss, rss <= GATES["peak_rss_max_bytes"]),
        "wer": status(wer, wer is not None and wer <= GATES["wer_max"]),
        "speaker_attribution": status(accuracy, accuracy is not None and accuracy >= GATES["speaker_accuracy_min"]),
    }


def check_label(_ctx: click.Context, _param: click.Parameter, label: str) -> str:
    if not _LABEL.fullmatch(label):
        msg = f"{label!r} is not a plain name: use letters, digits, '.', '_' and '-', starting with a letter or digit"
        raise click.BadParameter(msg)
    return label


def write_new_report(directory: Path, stem: str, content: str) -> Path:
    """Create a new report file, adding a counter instead of replacing an existing report."""
    for attempt in range(1000):
        destination = directory / (f"{stem}.json" if attempt == 0 else f"{stem}-{attempt}.json")
        try:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(content)
        except FileExistsError:
            continue
        return destination
    msg = f"too many reports named {stem}"
    raise click.ClickException(msg)


@click.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--label",
    required=True,
    callback=check_label,
    help="Name for the report (letters, digits, '.', '_', '-'); never derived from the file name.",
)
@click.option("--reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--alignment-out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write the word alignment (contains recognized text) to check the reference by hand.",
)
@click.option("--set", "overrides", multiple=True, metavar="FIELD=VALUE", help="Override a Settings field.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("bench/reports"),
    show_default=True,
)
def main(
    audio: Path,
    label: str,
    reference: Path | None,
    alignment_out: Path | None,
    overrides: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Benchmark one recording and write bench/reports/LABEL-TIMESTAMP.json."""
    settings = apply_overrides(overrides)
    started = time.perf_counter()
    transcript = build_transcript(audio, settings)
    total = time.perf_counter() - started
    words = transcript.words
    report: dict[str, Any] = {
        "label": label,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": host_info(),
        "models": model_hashes(settings),
        "settings": settings_report(settings),
        "audio_seconds": round(transcript.audio_seconds, 2),
        "stage_seconds": {name: round(value, 2) for name, value in transcript.stage_seconds.items()},
        "total_seconds": round(total, 2),
        "real_time_factor": round(total / transcript.audio_seconds, 4) if transcript.audio_seconds else None,
        "peak_rss_bytes": peak_rss_bytes(),
        "speakers_found": transcript.speakers_found,
        "words": len(words),
        "paragraphs": len(transcript.paragraphs),
        "unknown_share": round(sum(w.speaker is None for w in words) / len(words), 4) if words else None,
        "ambiguous_share": round(sum(w.ambiguous for w in words) / len(words), 4) if words else None,
    }
    if reference is not None:
        report.update(score(reference, words, alignment_out))
    report["gates"] = gates(report)
    report["gate_targets"] = GATES

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = write_new_report(output_dir, f"{label}-{stamp}", json.dumps(report, indent=2) + "\n")
    summary = {key: report.get(key) for key in ("real_time_factor", "peak_rss_bytes", "speakers_found")}
    summary["wer"] = report.get("wer", {}).get("wer")
    summary["speaker_accuracy"] = report.get("speaker_accuracy", {}).get("one_to_one")
    click.echo(f"{label}: {json.dumps(summary)} gates={json.dumps(report['gates'])}", err=True)
    click.echo(f"report: {destination}", err=True)


if __name__ == "__main__":
    main()
