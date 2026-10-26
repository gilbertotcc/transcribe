"""Developer tool: diarization experiments on cached ASR, without touching the shipped pipeline.

    uv run scripts/diarization_lab.py prepare AUDIO --label L [--reference REF.tsv | --rttm REF.rttm]
    uv run scripts/diarization_lab.py diarize --label L [--threshold T] [--num-clusters K] [--embedding MODEL]
    uv run scripts/diarization_lab.py embed --label L --turns ID [--embedding MODEL]
    uv run scripts/diarization_lab.py sweep --label L --turns ID [--embedding MODEL] --method cluster,turn ...
    uv run scripts/diarization_lab.py score --label L --turns ID [ID ...]
    uv run scripts/diarization_lab.py diagnose-short --label L --turns ID

`prepare` decodes the audio and, with a text reference, runs language detection
and ASR once. Every later command reuses that cache, so a setting costs one
diarization run (`diarize`) or a few seconds (`sweep`, which re-clusters turns
from their embeddings). Everything is written under the git-ignored
`bench/lab/LABEL/`; the cached words hold recognized text, and the ledger
`bench/lab/ledger.jsonl` holds metrics and settings only.

Re-clustering (`recluster_turns`) works on the turns of one sherpa-onnx run:
turns of at least `min_seconds` are anchors; `cluster` merges sherpa's clusters
by the average-linkage distance of their anchor centroids, `turn` ignores
sherpa's labels and clusters the anchor turns directly. Turns without a usable
anchor join the nearest group when their cosine similarity reaches `reject`.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from dataclasses import dataclass
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

from transcribe.attribution import attribute
from transcribe.audio import decode
from transcribe.domain import AttributedWord, Turn, Word, clamp_interval
from transcribe.models import sha256_file
from transcribe.settings import Settings


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    Vectors = npt.NDArray[np.float64]

LAB = Path("bench/lab")
EDGE_SECONDS = 0.1  # trimmed from each end of a turn before embedding it, away from neighbouring speech
MIN_EMBED_SECONDS = 0.25  # shorter (trimmed) turns get no embedding
NEAR_SECONDS = 0.5  # diagnose-short: a correct-speaker turn this close counts as a boundary error


# --- Clustering (pure) --------------------------------------------------------


def normalize(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def average_linkage(vectors: Vectors, weights: Vectors, *, threshold: float, k: int | None = None) -> list[int]:
    """Group unit vectors by weighted average-linkage cosine distance.

    Merging stops when the closest groups are farther apart than `threshold`,
    or, when `k` is given, at `k` groups regardless of distance.
    Returns a group index per row, numbered by first row.
    """
    n = len(vectors)
    if n == 0:
        return []
    distance = 1.0 - vectors @ vectors.T
    np.fill_diagonal(distance, np.inf)
    weight = weights.astype(np.float64).copy()
    group = np.arange(n)
    groups = n
    while groups > 1 and (k is None or groups > k):
        i, j = divmod(int(np.argmin(distance)), n)
        if k is None and distance[i, j] > threshold:
            break
        merged = (weight[i] * distance[i] + weight[j] * distance[j]) / (weight[i] + weight[j])
        distance[i, :] = merged
        distance[:, i] = merged
        distance[i, i] = np.inf
        distance[j, :] = np.inf
        distance[:, j] = np.inf
        weight[i] += weight[j]
        group[group == j] = i
        groups -= 1
    numbering: dict[int, int] = {}
    return [numbering.setdefault(int(g), len(numbering)) for g in group]


def _centroid(vectors: Vectors, weights: Vectors) -> Vectors:
    centroid: Vectors = normalize((weights[:, None] * vectors).sum(axis=0, keepdims=True))[0]
    return centroid


@dataclass(frozen=True, slots=True)
class Recluster:
    method: str  # "cluster" (merge sherpa's clusters) or "turn" (cluster anchor turns)
    threshold: float  # average-linkage cosine distance
    min_seconds: float  # anchor turns are at least this long
    reject: float  # minimum cosine similarity to join a group without an anchor
    k: int | None = None  # oracle group count, diagnostics only
    fold_seconds: float = 0.0  # groups with less speech than this fold into the nearest larger group...
    fold_similarity: float = 0.0  # ...when their cosine similarity reaches this

    @property
    def name(self) -> str:
        k = "auto" if self.k is None else str(self.k)
        fold = f"-f{self.fold_seconds:g}s{self.fold_similarity:g}" if self.fold_seconds > 0 else ""
        return f"{self.method}-t{self.threshold:g}-m{self.min_seconds:g}-r{self.reject:g}-k{k}{fold}"


def recluster_turns(turns: Sequence[Turn], embeddings: Vectors, config: Recluster) -> list[Turn]:
    """Relabel turns; `embeddings` has one unit row per turn, all zeros where a turn has none."""
    if not turns:
        return []
    durations = np.array([turn.end - turn.start for turn in turns])
    has_vector = np.linalg.norm(embeddings, axis=1) > 0
    anchor = has_vector & (durations >= config.min_seconds)
    # Units to group: sherpa's clusters, or single anchor turns. Each unit is a centroid with a weight.
    if config.method == "cluster":
        unit_of_turn = [turn.speaker for turn in turns]
    elif config.method == "turn":
        unit_of_turn = [index if anchor[index] else -1 - turn.speaker for index, turn in enumerate(turns)]
    else:
        msg = f"unknown re-clustering method {config.method!r}"
        raise ValueError(msg)
    units = sorted(set(unit_of_turn))
    anchored = [u for u in units if any(anchor[i] for i, x in enumerate(unit_of_turn) if x == u)]
    members = {u: [i for i, x in enumerate(unit_of_turn) if x == u] for u in units}

    def centroid(indices: Sequence[int]) -> Vectors:
        rows = list(indices)
        return _centroid(embeddings[rows], durations[rows])

    anchor_centroids = np.array([centroid([i for i in members[u] if anchor[i]]) for u in anchored])
    anchor_weights = np.array([sum(durations[i] for i in members[u] if anchor[i]) for u in anchored])
    group_of_anchored = average_linkage(anchor_centroids, anchor_weights, threshold=config.threshold, k=config.k)
    group_of_unit: dict[int, int] = dict(zip(anchored, group_of_anchored, strict=True))
    n_groups = max(group_of_anchored, default=-1) + 1
    group_centroids = [
        _centroid(
            anchor_centroids[[g == group for g in group_of_anchored]],
            anchor_weights[[g == group for g in group_of_anchored]],
        )
        for group in range(n_groups)
    ]
    # Units without anchors: join the nearest group if similar enough, else stay a group of their own.
    for unit in units:
        if unit in group_of_unit:
            continue
        with_vector = [i for i in members[unit] if has_vector[i]]
        best_group, best_similarity = -1, -np.inf
        if with_vector and group_centroids:
            similarities = np.array(group_centroids) @ centroid(with_vector)
            best_group, best_similarity = int(np.argmax(similarities)), float(np.max(similarities))
        if best_similarity >= config.reject:
            group_of_unit[unit] = best_group
        else:
            group_of_unit[unit] = n_groups
            n_groups += 1
    labels = [group_of_unit[unit_of_turn[i]] for i in range(len(turns))]
    if config.fold_seconds > 0:
        labels = fold_small_groups(labels, embeddings, durations, config.fold_seconds, config.fold_similarity)
    return [Turn(turn.start, turn.end, labels[i]) for i, turn in enumerate(turns)]


def fold_small_groups(
    labels: Sequence[int], embeddings: Vectors, durations: Vectors, min_seconds: float, similarity: float
) -> list[int]:
    """Relabel groups with less than `min_seconds` of speech as their most similar larger group, if similar enough."""
    groups = sorted(set(labels))
    rows = {g: [i for i, label in enumerate(labels) if label == g] for g in groups}
    seconds = {g: float(durations[rows[g]].sum()) for g in groups}
    with_vector = {g: [i for i in rows[g] if np.linalg.norm(embeddings[i]) > 0] for g in groups}
    large = [g for g in groups if seconds[g] >= min_seconds and with_vector[g]]
    if not large:
        return list(labels)
    centroids = np.array([_centroid(embeddings[with_vector[g]], durations[with_vector[g]]) for g in large])
    target = {}
    for g in groups:
        if g in large or not with_vector[g]:
            continue
        similarities = centroids @ _centroid(embeddings[with_vector[g]], durations[with_vector[g]])
        if float(np.max(similarities)) >= similarity:
            target[g] = large[int(np.argmax(similarities))]
    return [target.get(label, label) for label in labels]


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


def embedding_path(settings_or_path: str | None) -> Path:
    if settings_or_path is None:
        return Settings().embedding_model_path
    path = Path(settings_or_path).expanduser()
    return path if path.is_absolute() or path.is_file() else Settings().models_root / settings_or_path


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
        if self.words:
            accuracy, _ = speaker_accuracy(self.triples(attribute(self.words, turns)))
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
    from transcribe.asr import detect_language, transcribe  # engines load lazily

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
@click.option("--threshold", type=float, default=Settings().clustering_threshold, show_default=True)
@click.option("--num-clusters", type=int, default=-1, show_default=True, help="-1 infers the count.")
@click.option("--embedding", help="Embedding model: a path, or a file under the models root.")
def diarize(label: str, threshold: float, num_clusters: int, embedding: str | None) -> None:
    """Run sherpa-onnx diarization once, as the shipped adapter does, and cache its turns."""
    import sherpa_onnx  # engines load lazily

    directory = lab_dir(label)
    settings = Settings()
    model = embedding_path(embedding)
    samples = np.load(directory / "audio.npy")
    started = time.perf_counter()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(settings.segmentation_model_path),
                window_shift_ratio=settings.window_shift_ratio,
            ),
            num_threads=settings.diarization_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=settings.diarization_threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=settings.min_duration_on,
        min_duration_off=settings.min_duration_off,
    )
    if not config.validate():
        msg = "invalid diarization configuration"
        raise click.ClickException(msg)
    result = sherpa_onnx.OfflineSpeakerDiarization(config).process(samples).sort_by_start_time()
    elapsed = time.perf_counter() - started
    duration = len(samples) / settings.sample_rate
    turns: list[Turn] = []
    for segment in result:
        start, end = clamp_interval(segment.start, segment.end, duration)
        turns.append(Turn(start, end, int(segment.speaker)))
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
    """One unit embedding per turn (zeros when too short), cached per turns and model."""
    import sherpa_onnx  # engines load lazily

    cache = directory / "embeddings" / f"{turns_id}__{model.stem}.npy"
    if cache.is_file():
        cached: Vectors = np.load(cache)
        return cached
    settings = Settings()
    samples = np.load(directory / "audio.npy")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=settings.diarization_threads)
    )
    turns = load_turns(directory, turns_id)
    vectors = np.zeros((len(turns), extractor.dim), dtype=np.float64)
    started = time.perf_counter()
    for index, turn in enumerate(turns):
        start, end = turn.start + EDGE_SECONDS, turn.end - EDGE_SECONDS
        if end - start < MIN_EMBED_SECONDS:
            start, end = turn.start, turn.end
        if end - start < MIN_EMBED_SECONDS:
            continue
        stream = extractor.create_stream()
        rate = settings.sample_rate
        stream.accept_waveform(sample_rate=rate, waveform=samples[int(start * rate) : int(end * rate)])
        stream.input_finished()
        vectors[index] = np.array(extractor.compute(stream), dtype=np.float64)
    vectors = normalize(vectors)
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
    model = embedding_path(embedding)
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
@click.option("--method", default="cluster", show_default=True, help="Comma-separated: cluster, turn.")
@click.option("--threshold", "thresholds", required=True, help="Comma-separated cosine distances.")
@click.option("--min-seconds", default="1.5", show_default=True, help="Comma-separated anchor lengths.")
@click.option("--reject", default="0.5", show_default=True, help="Comma-separated cosine similarities.")
@click.option("--k", type=int, help="Oracle group count (diagnostics only; ignores --threshold).")
@click.option("--fold-seconds", default="0", show_default=True, help="Comma-separated; 0 folds nothing.")
@click.option("--fold-similarity", default="0.3", show_default=True, help="Comma-separated cosine similarities.")
@click.option("--save", is_flag=True, help="Also cache the relabeled turns, for diagnose-short.")
def sweep(
    label: str,
    turns_id: str,
    embedding: str | None,
    method: str,
    thresholds: str,
    min_seconds: str,
    reject: str,
    k: int | None,
    fold_seconds: str,
    fold_similarity: str,
    save: bool,  # noqa: FBT001 - Click passes flags positionally by name
) -> None:
    """Re-cluster one cached sherpa run over a grid of settings and score each."""
    directory = lab_dir(label)
    model = embedding_path(embedding)
    turns = load_turns(directory, turns_id)
    vectors = embeddings_for(directory, turns_id, model)
    scorer = Scorer(directory)
    grid = itertools.product(
        method.split(","),
        _floats(thresholds),
        _floats(min_seconds),
        _floats(reject),
        _floats(fold_seconds),
        _floats(fold_similarity),
    )
    for method_name, threshold, anchor_seconds, similarity, fold, fold_min in grid:
        config = Recluster(method_name, threshold, anchor_seconds, similarity, k, fold, fold_min)
        started = time.perf_counter()
        relabeled = recluster_turns(turns, vectors, config)
        elapsed = time.perf_counter() - started
        metrics = scorer.score(relabeled)
        name = f"{turns_id}+{config.name}-{model.stem}"
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
                    "method": method_name,
                    "threshold": threshold,
                    "min_seconds": anchor_seconds,
                    "reject": similarity,
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
    _, mapping = speaker_accuracy(scorer.triples(attributed))
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
