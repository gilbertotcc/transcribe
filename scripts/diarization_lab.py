"""Developer tool: diarization experiments on cached ASR, without touching the shipped pipeline.

    uv run scripts/diarization_lab.py prepare AUDIO --label L [--reference REF.tsv | --rttm REF.rttm]
    uv run scripts/diarization_lab.py diarize --label L [--threshold T] [--num-clusters K] [--embedding MODEL]
    uv run scripts/diarization_lab.py embed --label L --turns ID [--embedding MODEL]
    uv run scripts/diarization_lab.py sweep --label L --turns ID [--embedding MODEL] [--threshold 0.35,0.4 ...]
    uv run scripts/diarization_lab.py score --label L --turns ID [ID ...]
    uv run scripts/diarization_lab.py diagnose-short --label L --turns ID

`prepare` decodes the audio and, with a text reference, runs language detection
and ASR once. Every later command reuses that cache, so a setting costs one
sherpa-onnx run (`diarize`, without re-clustering) or a few seconds (`sweep`,
which re-clusters cached turns with
`transcribe.pipeline.steps.diarization.clustering.recluster`, as the pipeline
does; its defaults are the `Settings` defaults). Everything is written
under the git-ignored `bench/lab/LABEL/`; the cached words hold recognized text,
and the ledger `bench/lab/ledger.jsonl` holds metrics and settings only.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from benchmark import (
    DER_COLLARS,
    SHORT_TURN_MAX,
    align,
    check_label,
    hypothesis,
    peak_rss_bytes,
    read_rttm,
    score_turns,
    speaker_accuracy,
)

from transcribe.config import Settings
from transcribe.domain import AttributedWord, Turn, Word
from transcribe.models import sha256_file
from transcribe.pipeline.steps.attribution import attribute
from transcribe.pipeline.steps.decode import decode
from transcribe.pipeline.steps.diarization.clustering import recluster
from transcribe.pipeline.steps.diarization.sherpa import embed_turns, sherpa_turns


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    Vectors = npt.NDArray[np.float64]

LAB = Path("bench/lab")
DEFAULTS = Settings()
NEAR_SECONDS = 0.5  # diagnose-short: a correct-speaker turn this close counts as a boundary error


def recluster_name(threshold: float, anchor: float, join: float, k: int | None, fold: float, fold_min: float) -> str:
    """Cache name of one re-clustering setting (the `turn-` prefix keeps earlier cache names valid)."""
    folding = f"-f{fold:g}s{fold_min:g}" if fold > 0 else ""
    return f"turn-t{threshold:g}-m{anchor:g}-r{join:g}-k{'auto' if k is None else k}{folding}"


# --- Cache --------------------------------------------------------------------


def lab_dir(label: str) -> Path:
    path = LAB / label
    if not (path / "meta.json").is_file():
        msg = f"no prepared recording named {label!r}; run `prepare` first"
        raise click.ClickException(msg)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")


def load_turns(directory: Path, turns_id: str) -> list[Turn]:
    path = directory / "turns" / f"{turns_id}.json"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in (directory / "turns").glob("*.json")))
        msg = f"no turns {turns_id!r}; choose from {known}"
        raise click.ClickException(msg)
    return [Turn(start, end, speaker) for start, end, speaker in read_json(path)["turns"]]


def load_words(directory: Path) -> list[Word]:
    return [Word(start, end, text) for start, end, text in read_json(directory / "words.json")]


def embedding_path(name: str | None, default: Path) -> Path:
    """A model path as given, or a file under the models root; `default` when not given."""
    if name is None:
        return default
    path = Path(name).expanduser()
    return path if path.is_absolute() or path.is_file() else DEFAULTS.models_root / name


RECLUSTER_MODEL = DEFAULTS.recluster_embedding_model_path or DEFAULTS.embedding_model_path


def revision() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)  # noqa: S607
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)  # noqa: S607
    return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def ledger(entry: dict[str, Any]) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    line = {"created": datetime.now(UTC).isoformat(timespec="seconds"), **revision(), **entry}
    with (LAB / "ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(line) + "\n")


# --- Scoring ------------------------------------------------------------------


class Scorer:
    """Scores turn sets of one recording; the word alignment is computed once."""

    def __init__(self, directory: Path) -> None:
        self.meta: dict[str, Any] = read_json(directory / "meta.json")
        self.words = load_words(directory) if self.meta.get("reference") else []
        self.rttm = read_rttm(Path(self.meta["rttm"])) if self.meta.get("rttm") else None
        if self.words:
            placeholder = [AttributedWord(word, None, ambiguous=False) for word in self.words]
            tokens, _ = hypothesis(placeholder)
            self.token_word: list[int] = []
            for index, item in enumerate(placeholder):
                count = len(hypothesis([item])[0])
                self.token_word.extend([index] * count)
            assert len(self.token_word) == len(tokens)  # noqa: S101 - internal invariant
            self.alignment = align(Path(self.meta["reference"]), tokens)

    def triples(self, attributed: Sequence[AttributedWord]) -> list[tuple[int | None, str, int]]:
        return [
            (attributed[self.token_word[index]].speaker, reader, length)
            for index, reader, length in self.alignment.scored
        ]

    def score(self, turns: Sequence[Turn]) -> dict[str, Any]:
        result: dict[str, Any] = {"speakers_found": len({turn.speaker for turn in turns})}
        triples = self.triples(attribute(self.words, turns)) if self.words else []
        if triples:
            accuracy, _ = speaker_accuracy(triples)
            short = accuracy["by_turn_length"].get(f"turns_1_{SHORT_TURN_MAX}_words", {})
            result |= {
                "one_to_one": accuracy["one_to_one"],
                "many_to_one": accuracy["many_to_one"],
                "short_one_to_one": short.get("one_to_one"),
                "short_many_to_one": short.get("many_to_one"),
                "scored_words": accuracy["scored_words"],
            }
        if self.rttm is not None:
            turns_report = score_turns(self.rttm, turns, self.meta["audio_seconds"])
            result |= {
                "der": turns_report["der"][f"collar_{DER_COLLARS[-1]}"]["der"],
                "der_0": turns_report["der"][f"collar_{DER_COLLARS[0]}"]["der"],
                "jer": turns_report["jer"],
                "purity": turns_report["purity"],
                "coverage": turns_report["coverage"],
            }
        return result


# --- Commands -----------------------------------------------------------------


@click.group()
def main() -> None:
    """Diarization experiments on cached audio and words (bench/lab/)."""


@main.command()
@click.argument("audio", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--label", required=True, callback=check_label)
@click.option("--reference", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="SPEAKER<TAB>text.")
@click.option("--rttm", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Time-stamped reference.")
def prepare(audio: Path, label: str, reference: Path | None, rttm: Path | None) -> None:
    """Decode AUDIO once and, with --reference, run language detection and ASR once."""
    from transcribe.pipeline.steps.asr.whisper_ct2 import transcribe  # engines load lazily
    from transcribe.pipeline.steps.language import detect_language

    settings = Settings()
    directory = LAB / label
    directory.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = {}
    started = time.perf_counter()
    samples = decode(audio, settings)
    seconds["decode"] = time.perf_counter() - started
    np.save(directory / "audio.npy", samples)
    language = None
    if reference is not None:
        started = time.perf_counter()
        detected = detect_language(samples, settings)
        seconds["language_detection"] = time.perf_counter() - started
        if detected is None:
            msg = "no speech found"
            raise click.ClickException(msg)
        language = detected.code
        started = time.perf_counter()
        words = transcribe(samples, settings, language)
        seconds["asr"] = time.perf_counter() - started
        write_json(directory / "words.json", [[word.start, word.end, word.text] for word in words])
    meta = {
        "audio_sha256": sha256_file(audio),
        "audio_seconds": len(samples) / settings.sample_rate,
        "reference": None if reference is None else str(reference.resolve()),
        "rttm": None if rttm is None else str(rttm.resolve()),
        "language": language,
        "stage_seconds": {name: round(value, 2) for name, value in seconds.items()},
        "peak_rss_bytes": peak_rss_bytes(),
    }
    write_json(directory / "meta.json", meta)
    click.echo(f"{label}: prepared, {json.dumps(meta['stage_seconds'])}", err=True)


@main.command()
@click.option("--label", required=True, callback=check_label)
@click.option("--threshold", type=float, default=DEFAULTS.clustering_threshold, show_default=True)
@click.option("--num-clusters", type=int, default=-1, show_default=True, help="-1 infers the count.")
@click.option("--embedding", help="sherpa-onnx's embedding model: a path, or a file under the models root.")
def diarize(label: str, threshold: float, num_clusters: int, embedding: str | None) -> None:
    """Run sherpa-onnx diarization once, as the shipped adapter does before re-clustering, and cache its turns."""
    directory = lab_dir(label)
    model = embedding_path(embedding, DEFAULTS.embedding_model_path)
    settings = dataclasses.replace(DEFAULTS, clustering_threshold=threshold, embedding_model=str(model))
    samples = np.load(directory / "audio.npy")
    started = time.perf_counter()
    turns = sherpa_turns(samples, settings, num_clusters=num_clusters)
    elapsed = time.perf_counter() - started
    duration = len(samples) / settings.sample_rate
    k = "auto" if num_clusters < 0 else str(num_clusters)
    turns_id = f"sherpa-t{threshold:g}-k{k}-{model.stem}"
    run = {
        "config": {"threshold": threshold, "num_clusters": num_clusters, "embedding": model.name},
        "embedding_sha256": sha256_file(model),
        "seconds": round(elapsed, 2),
        "real_time_factor": round(elapsed / duration, 4),
        "peak_rss_bytes": peak_rss_bytes(),
        "turns": [[turn.start, turn.end, turn.speaker] for turn in turns],
    }
    write_json(directory / "turns" / f"{turns_id}.json", run)
    metrics = Scorer(directory).score(turns)
    ledger(
        {
            "label": label,
            "command": "diarize",
            "turns": turns_id,
            **{k: v for k, v in run.items() if k != "turns"},
            **metrics,
        }
    )
    click.echo(f"{label} {turns_id}: {elapsed:.1f} s, {json.dumps(metrics)}", err=True)


def embeddings_for(directory: Path, turns_id: str, model: Path) -> Vectors:
    """The pipeline's turn embeddings, cached per turns and model."""
    cache = directory / "embeddings" / f"{turns_id}__{model.stem}.npy"
    if cache.is_file():
        cached: Vectors = np.load(cache)
        return cached
    turns = load_turns(directory, turns_id)
    started = time.perf_counter()
    vectors = embed_turns(np.load(directory / "audio.npy"), turns, model, DEFAULTS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    click.echo(f"embedded {len(turns)} turns with {model.name} in {time.perf_counter() - started:.1f} s", err=True)
    return vectors


@main.command()
@click.option("--label", required=True, callback=check_label)
@click.option("--turns", "turns_id", required=True)
@click.option("--embedding", help="Embedding model: a path, or a file under the models root.")
def embed(label: str, turns_id: str, embedding: str | None) -> None:
    """Embed every turn of a cached sherpa run (timed; cached for `sweep`)."""
    directory = lab_dir(label)
    model = embedding_path(embedding, RECLUSTER_MODEL)
    started = time.perf_counter()
    embeddings_for(directory, turns_id, model)
    ledger(
        {
            "label": label,
            "command": "embed",
            "turns": turns_id,
            "embedding": model.name,
            "embedding_sha256": sha256_file(model),
            "seconds": round(time.perf_counter() - started, 2),
            "peak_rss_bytes": peak_rss_bytes(),
        }
    )


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


@main.command()
@click.option("--label", required=True, callback=check_label)
@click.option("--turns", "turns_id", required=True)
@click.option("--embedding", help="Embedding model: a path, or a file under the models root.")
@click.option("--threshold", "thresholds", default=str(DEFAULTS.recluster_threshold), show_default=True)
@click.option("--anchor-seconds", default=str(DEFAULTS.recluster_anchor_seconds), show_default=True)
@click.option("--join-similarity", default=str(DEFAULTS.recluster_join_similarity), show_default=True)
@click.option("--fold-seconds", default=str(DEFAULTS.recluster_fold_seconds), show_default=True, help="0: no folding.")
@click.option("--fold-similarity", default=str(DEFAULTS.recluster_fold_similarity), show_default=True)
@click.option("--k", type=int, help="Oracle group count (diagnostics only; ignores --threshold).")
@click.option("--save", is_flag=True, help="Also cache the relabeled turns, for diagnose-short.")
def sweep(
    label: str,
    turns_id: str,
    embedding: str | None,
    thresholds: str,
    anchor_seconds: str,
    join_similarity: str,
    fold_seconds: str,
    fold_similarity: str,
    k: int | None,
    save: bool,  # noqa: FBT001 - Click passes flags positionally by name
) -> None:
    """Re-cluster one cached sherpa run over a grid of comma-separated settings and score each."""
    directory = lab_dir(label)
    model = embedding_path(embedding, RECLUSTER_MODEL)
    turns = load_turns(directory, turns_id)
    vectors = embeddings_for(directory, turns_id, model)
    scorer = Scorer(directory)
    grid = itertools.product(
        _floats(thresholds),
        _floats(anchor_seconds),
        _floats(join_similarity),
        _floats(fold_seconds),
        _floats(fold_similarity),
    )
    for threshold, anchor, join, fold, fold_min in grid:
        started = time.perf_counter()
        relabeled = recluster(
            turns,
            vectors,
            threshold=threshold,
            anchor_seconds=anchor,
            join_similarity=join,
            fold_seconds=fold,
            fold_similarity=fold_min,
            k=k,
        )
        elapsed = time.perf_counter() - started
        metrics = scorer.score(relabeled)
        name = f"{turns_id}+{recluster_name(threshold, anchor, join, k, fold, fold_min)}-{model.stem}"
        if save:
            write_json(
                directory / "turns" / f"{name}.json", {"turns": [[t.start, t.end, t.speaker] for t in relabeled]}
            )
        ledger(
            {
                "label": label,
                "command": "sweep",
                "turns": name,
                "base": turns_id,
                "embedding": model.name,
                "recluster": {
                    "threshold": threshold,
                    "anchor_seconds": anchor,
                    "join_similarity": join,
                    "k": k,
                    "fold_seconds": fold,
                    "fold_similarity": fold_min,
                },
                "recluster_seconds": round(elapsed, 3),
                **metrics,
            }
        )
        click.echo(f"{label} {name}: {json.dumps(metrics)}", err=True)


@main.command()
@click.option("--label", required=True, callback=check_label)
@click.option("--turns", "turns_ids", required=True, multiple=True)
def score(label: str, turns_ids: tuple[str, ...]) -> None:
    """Score cached turn sets."""
    directory = lab_dir(label)
    scorer = Scorer(directory)
    for turns_id in turns_ids:
        click.echo(f"{label} {turns_id}: {json.dumps(scorer.score(load_turns(directory, turns_id)))}", err=True)


def short_reply_causes(
    words: Sequence[Word], turns: Sequence[Turn], scored: Sequence[tuple[int, int | None, str]], mapping: dict[int, str]
) -> dict[str, int]:
    """Why short-reply words are wrong: (word index, assigned cluster, reader) under the one-to-one map."""
    causes = {"correct": 0, "no_turn": 0, "unmapped_cluster": 0, "near_correct_turn": 0, "no_correct_turn_near": 0}
    for index, cluster, reader in scored:
        if cluster is not None and mapping.get(cluster) == reader:
            causes["correct"] += 1
        elif cluster is None:
            causes["no_turn"] += 1
        elif cluster not in mapping:
            causes["unmapped_cluster"] += 1
        else:
            word = words[index]
            near = any(
                mapping.get(turn.speaker) == reader
                and turn.start < word.end + NEAR_SECONDS
                and turn.end > word.start - NEAR_SECONDS
                for turn in turns
            )
            causes["near_correct_turn" if near else "no_correct_turn_near"] += 1
    return causes


@main.command("diagnose-short")
@click.option("--label", required=True, callback=check_label)
@click.option("--turns", "turns_id", required=True)
def diagnose_short(label: str, turns_id: str) -> None:
    """Count why words of short replies get the wrong speaker (counts only, no text)."""
    directory = lab_dir(label)
    scorer = Scorer(directory)
    if not scorer.words:
        msg = "diagnose-short needs a recording prepared with --reference"
        raise click.ClickException(msg)
    turns = load_turns(directory, turns_id)
    attributed = attribute(scorer.words, turns)
    triples = scorer.triples(attributed)
    if not triples:
        msg = "diagnose-short found no speaker-scored reference words"
        raise click.ClickException(msg)
    _, mapping = speaker_accuracy(triples)
    seen: set[int] = set()
    scored: list[tuple[int, int | None, str]] = []
    for token, reader, length in scorer.alignment.scored:
        index = scorer.token_word[token]
        if length <= SHORT_TURN_MAX and index not in seen:
            seen.add(index)
            scored.append((index, attributed[index].speaker, reader))
    causes = short_reply_causes(scorer.words, turns, scored, mapping)
    ledger({"label": label, "command": "diagnose-short", "turns": turns_id, "causes": causes})
    click.echo(f"{label} {turns_id}: {json.dumps(causes)}", err=True)


if __name__ == "__main__":
    main()
