"""Developer tool: run the pipeline on one recording and write a JSON report.

    uv run scripts/benchmark.py AUDIO --label NAME [--reference REF] [--set field=value ...]

The report holds measurements only: no recognized text and no file paths. Run
each measurement in a fresh process (peak RSS is per process).

REF is plain text, or `SPEAKER<TAB>text` lines (see scripts/play_reference.py)
to also score speaker attribution; `-` marks text without a known speaker.
The reference may cover more than the recording: both sides are trimmed to the
span where they align, and the trimmed word counts are reported.

`--diarization-only` skips speech recognition and scores the speaker turns in
time (DER, JER, purity, coverage) against an RTTM reference
(scripts/simulate_conversation.py), or only counts clusters against
`--speakers-known N` when no RTTM exists:

    uv run scripts/benchmark.py AUDIO --label NAME --diarization-only (--rttm REF.rttm | --speakers-known N)
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
import unicodedata
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import jiwer
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import (
    DiarizationCoverage,
    DiarizationErrorRate,
    DiarizationPurity,
    JaccardErrorRate,
)

from transcribe.application import build_transcript
from transcribe.audio import decode
from transcribe.diarization import diarize
from transcribe.models import PROVISIONING_HINT, load_manifest, required_paths, sha256_file
from transcribe.settings import DEFAULT_MODELS_ROOT, Settings


if TYPE_CHECKING:
    from collections.abc import Sequence

    from transcribe.domain import AttributedWord, Turn

GATES = {"rtf_max": 1.0, "peak_rss_max_bytes": 6_000_000_000, "wer_max": 0.08, "speaker_accuracy_min": 0.90}
UNSCORED = "-"
SHORT_TURN_MAX = 2  # words; replies such as "Yes" or "That's great"
_DASHES = re.compile("[\u2014\u2013\\-/]")
_NOT_WORD = re.compile(r"[^\w' ]|_")  # Unicode letters and digits, so Italian "perché" stays one word
_MIN_ANCHOR = 4  # words in a matching block that anchors the alignment window
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # a plain file-name stem, so reports stay in --output-dir
DER_COLLARS = (0.0, 0.5)  # seconds, total width around each reference boundary (pyannote: 0.5 is +/-0.25 s)


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


def diarization_paths(settings: Settings) -> tuple[Path, ...]:
    return (settings.segmentation_model_path, settings.embedding_model_path)


def model_hashes(settings: Settings, paths: Sequence[Path] | None = None) -> dict[str, str | None]:
    """Manifest hash of each model file; files not in the manifest (candidates) are hashed.

    Files outside the models root are listed by name, never by full path.
    """
    manifest = {settings.models_root / f.path: f.sha256 for asset in load_manifest() for f in asset.files}
    hashes: dict[str, str | None] = {}
    for path in required_paths(settings) if paths is None else paths:
        key = str(path.relative_to(settings.models_root)) if path.is_relative_to(settings.models_root) else path.name
        if path in manifest:
            hashes[key] = manifest[path]
        else:
            hashes[key] = sha256_file(path) if path.is_file() else None
    return hashes


def apply_overrides(items: Sequence[str]) -> Settings:
    fields = {field.name: field for field in dataclasses.fields(Settings)}
    changes: dict[str, Any] = {}
    for item in items:
        name, _, raw = item.partition("=")
        if name not in fields:
            msg = f"unknown setting {name!r}; choose from {', '.join(fields)}"
            raise click.BadParameter(msg)
        kind = fields[name].type
        if kind == str | None:
            changes[name] = None if raw.lower() in {"", "none"} else raw
        elif kind is bool:
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


def _non_word(match: re.Match[str]) -> str:
    """A space, except for combining marks (Devanagari vowel signs, viramas), which belong to their word."""
    return match.group() if unicodedata.category(match.group()).startswith("M") else " "


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower().replace("\u2019", "'").replace("\u2018", "'")
    text = _DASHES.sub(" ", text)
    return [token.strip("'") for token in _NOT_WORD.sub(_non_word, text).split() if token.strip("'")]


def read_reference(path: Path) -> tuple[list[str], list[str], list[int]]:
    """Reference words, their speakers, and the word count of each word's line (one line is one turn)."""
    words: list[str] = []
    speakers: list[str] = []
    turn_lengths: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        speaker, tab, text = line.partition("\t")
        if not tab:
            speaker, text = UNSCORED, line
        line_tokens = tokens(text)
        words.extend(line_tokens)
        speakers.extend([speaker] * len(line_tokens))
        turn_lengths.extend([len(line_tokens)] * len(line_tokens))
    return words, speakers, turn_lengths


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


def best_one_to_one(
    counts: dict[tuple[int, str], int], clusters: list[int], readers: list[str]
) -> tuple[int, dict[int, str]]:
    """Exact maximum of correctly attributed words over one-to-one cluster->reader maps, and that map."""
    reader_bit = {reader: 1 << index for index, reader in enumerate(readers)}
    # used-readers mask -> (best score, cluster->reader map reaching it)
    best: dict[int, tuple[int, dict[int, str]]] = {0: (0, {})}
    for cluster in clusters:
        nxt = dict(best)
        for mask, (score, mapping) in best.items():
            for reader, bit in reader_bit.items():
                if not mask & bit:
                    candidate = score + counts.get((cluster, reader), 0)
                    if candidate > nxt.get(mask | bit, (-1, {}))[0]:
                        nxt[mask | bit] = (candidate, {**mapping, cluster: reader})
        best = nxt
    return max(best.values(), key=lambda item: item[0])


def turn_length_breakdown(
    triples: Sequence[tuple[int | None, str, int]], one_to_one: dict[int, str], many_to_one: dict[int, str]
) -> dict[str, dict[str, Any]]:
    """Speaker scores, under the whole-recording maps, for words in short and in longer reference turns."""
    breakdown: dict[str, dict[str, Any]] = {}
    for name, short in ((f"turns_1_{SHORT_TURN_MAX}_words", True), (f"turns_{SHORT_TURN_MAX + 1}_plus_words", False)):
        pairs = [(cluster, reader) for cluster, reader, length in triples if (length <= SHORT_TURN_MAX) == short]
        if pairs:
            breakdown[name] = {
                "one_to_one": round(sum(one_to_one.get(c) == r for c, r in pairs if c is not None) / len(pairs), 4),
                "many_to_one": round(sum(many_to_one.get(c) == r for c, r in pairs if c is not None) / len(pairs), 4),
                "scored_words": len(pairs),
                "unknown_words": sum(1 for cluster, _ in pairs if cluster is None),
            }
    return breakdown


def score(reference: Path, words: Sequence[AttributedWord], alignment_out: Path | None) -> dict[str, Any]:
    ref_all, ref_speakers_all, ref_turns_all = read_reference(reference)
    hyp, hyp_speakers = hypothesis(words)
    ref_window, _ = aligned_window(ref_all, hyp)
    ref, ref_speakers, ref_turns = ref_all[ref_window], ref_speakers_all[ref_window], ref_turns_all[ref_window]
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
    triples: list[tuple[int | None, str, int]] = []  # (cluster, reader, reference turn length)
    for chunk in output.alignments[0]:
        if chunk.type in {"equal", "substitute"}:
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                reader = ref_speakers[chunk.ref_start_idx + offset]
                if reader != UNSCORED:
                    triples.append(
                        (hyp_speakers[chunk.hyp_start_idx + offset], reader, ref_turns[chunk.ref_start_idx + offset])
                    )
    pairs = [(cluster, reader) for cluster, reader, _ in triples]
    if pairs:
        counts: dict[tuple[int, str], int] = {}
        for cluster, reader in pairs:
            if cluster is not None:
                counts[cluster, reader] = counts.get((cluster, reader), 0) + 1
        clusters = sorted({cluster for cluster, _ in counts})
        readers = sorted({reader for _, reader in pairs})
        majority = {c: max((counts.get((c, r), 0), r) for r in readers)[1] for c in clusters}
        exact, mapping = best_one_to_one(counts, clusters, readers)
        report["speaker_accuracy"] = {
            "one_to_one": round(exact / len(pairs), 4),
            "many_to_one": round(sum(counts.get((c, r), 0) for c, r in majority.items()) / len(pairs), 4),
            "scored_words": len(pairs),
            "unknown_words": sum(1 for cluster, _ in pairs if cluster is None),
            "reference_speakers": len(readers),
            "clusters_on_scored_words": len(clusters),
            "by_turn_length": turn_length_breakdown(triples, mapping, majority),
        }
    return report


# --- Time-based diarization scoring ------------------------------------------


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, speaker) of every SPEAKER line; the file must describe one recording."""
    turns: list[tuple[float, float, str]] = []
    recordings: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER":
            recordings.add(fields[1])
            start, duration = float(fields[3]), float(fields[4])
            turns.append((start, start + duration, fields[7]))
    if len(recordings) > 1:
        msg = f"the RTTM holds {len(recordings)} recordings; score one recording per file"
        raise click.UsageError(msg)
    return turns


def _annotation(turns: Sequence[tuple[float, float, str | int]]) -> Annotation:
    annotation = Annotation()
    for index, (start, end, speaker) in enumerate(turns):
        if end > start:
            annotation[Segment(start, end), index] = speaker
    return annotation


def score_turns(
    reference: Sequence[tuple[float, float, str]], turns: Sequence[Turn], audio_seconds: float
) -> dict[str, Any]:
    """DER at each collar, JER, purity and coverage of the speaker turns over the whole recording."""
    ref = _annotation(reference)
    hyp = _annotation([(turn.start, turn.end, turn.speaker) for turn in turns])
    uem = Timeline([Segment(0.0, audio_seconds)])
    der: dict[str, dict[str, float]] = {}
    for collar in DER_COLLARS:
        detail = DiarizationErrorRate(collar=collar)(ref, hyp, uem=uem, detailed=True)
        der[f"collar_{collar}"] = {
            "der": round(detail["diarization error rate"], 4),
            "false_alarm_seconds": round(detail["false alarm"], 2),
            "missed_seconds": round(detail["missed detection"], 2),
            "confusion_seconds": round(detail["confusion"], 2),
            "reference_seconds": round(detail["total"], 2),
        }
    return {
        "der": der,
        "jer": round(JaccardErrorRate()(ref, hyp, uem=uem), 4),
        "purity": round(DiarizationPurity()(ref, hyp, uem=uem), 4),
        "coverage": round(DiarizationCoverage()(ref, hyp, uem=uem), 4),
        "reference_speakers": len(ref.labels()),
        "reference_turns": len(reference),
        "speakers_found": len(hyp.labels()),
    }


def diarize_file(audio: Path, settings: Settings) -> tuple[float, list[Turn], dict[str, float]]:
    """Decode and diarize only, with the pipeline's own adapters; no speech recognition."""
    missing = [path.name for path in diarization_paths(settings) if not path.is_file()]
    if missing:
        msg = f"missing model files: {', '.join(missing)}; {PROVISIONING_HINT}"
        raise click.ClickException(msg)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    samples = decode(audio, settings)
    timings["decode"] = time.perf_counter() - started
    started = time.perf_counter()
    turns = diarize(samples, settings)
    timings["diarization"] = time.perf_counter() - started
    return len(samples) / settings.sample_rate, turns, timings


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
@click.option(
    "--diarization-only",
    is_flag=True,
    help="Skip speech recognition; score speaker turns in time against --rttm, or count them for --speakers-known.",
)
@click.option("--rttm", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Time-stamped reference.")
@click.option("--speakers-known", type=click.IntRange(min=1), help="Speaker count, when no RTTM exists.")
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
    diarization_only: bool,  # noqa: FBT001 - Click passes flags positionally by name
    rttm: Path | None,
    speakers_known: int | None,
    overrides: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Benchmark one recording and write bench/reports/LABEL-TIMESTAMP.json."""
    if diarization_only and reference is not None:
        msg = "--reference needs speech recognition; use --rttm with --diarization-only"
        raise click.UsageError(msg)
    if diarization_only and (rttm is None) == (speakers_known is None):
        msg = "--diarization-only needs exactly one of --rttm and --speakers-known"
        raise click.UsageError(msg)
    if not diarization_only and (rttm is not None or speakers_known is not None):
        msg = "--rttm and --speakers-known need --diarization-only"
        raise click.UsageError(msg)
    settings = apply_overrides(overrides)
    report: dict[str, Any] = {
        "label": label,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "diarization-only" if diarization_only else "full",
        "host": host_info(),
        "models": model_hashes(settings, diarization_paths(settings) if diarization_only else None),
        "settings": settings_report(settings),
    }
    started = time.perf_counter()
    if diarization_only:
        audio_seconds, turns, stage_seconds = diarize_file(audio, settings)
        total = time.perf_counter() - started
        speakers_found = len({turn.speaker for turn in turns})
    else:
        transcript = build_transcript(audio, settings)
        total = time.perf_counter() - started
        audio_seconds, stage_seconds, speakers_found = (
            transcript.audio_seconds,
            transcript.stage_seconds,
            transcript.speakers_found,
        )
    report.update(
        {
            "audio_seconds": round(audio_seconds, 2),
            "stage_seconds": {name: round(value, 2) for name, value in stage_seconds.items()},
            "total_seconds": round(total, 2),
            "real_time_factor": round(total / audio_seconds, 4) if audio_seconds else None,
            "peak_rss_bytes": peak_rss_bytes(),
            "speakers_found": speakers_found,
        }
    )
    if diarization_only:
        if rttm is not None:
            report["diarization"] = score_turns(read_rttm(rttm), turns, audio_seconds)
        else:
            report["diarization"] = {"reference_speakers": speakers_known, "speakers_found": speakers_found}
    else:
        words = transcript.words
        language = transcript.language
        report.update(
            {
                "language": None if language is None else {"code": language.code, "probability": language.probability},
                "words": len(words),
                "paragraphs": len(transcript.paragraphs),
                "unknown_share": round(sum(w.speaker is None for w in words) / len(words), 4) if words else None,
                "ambiguous_share": round(sum(w.ambiguous for w in words) / len(words), 4) if words else None,
            }
        )
        if reference is not None:
            report.update(score(reference, words, alignment_out))
    report["gates"] = gates(report)
    report["gate_targets"] = GATES

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = write_new_report(output_dir, f"{label}-{stamp}", json.dumps(report, indent=2) + "\n")
    summary = {key: report.get(key) for key in ("real_time_factor", "peak_rss_bytes", "speakers_found")}
    summary["language"] = (report.get("language") or {}).get("code")
    summary["wer"] = report.get("wer", {}).get("wer")
    summary["speaker_accuracy"] = report.get("speaker_accuracy", {}).get("one_to_one")
    if "der" in report.get("diarization", {}):
        summary["der"] = report["diarization"]["der"][f"collar_{DER_COLLARS[-1]}"]["der"]
    click.echo(f"{label}: {json.dumps(summary)} gates={json.dumps(report['gates'])}", err=True)
    click.echo(f"report: {destination}", err=True)


if __name__ == "__main__":
    main()
