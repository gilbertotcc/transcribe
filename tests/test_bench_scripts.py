"""Developer tools in scripts/: time-based diarization scoring and the conversation simulator (no models)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from transcribe.domain import Turn, Word
from transcribe.models import sha256_file
from transcribe.settings import Settings


if TYPE_CHECKING:
    from types import ModuleType


SCRIPTS = Path(__file__).parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module while the class is created
    spec.loader.exec_module(module)
    return module


benchmark = load_script("benchmark")
simulate = load_script("simulate_conversation")

REFERENCE = [(0.0, 10.0, "a"), (10.0, 20.0, "b")]


def test_perfect_turns_score_zero_der() -> None:
    result = benchmark.score_turns(REFERENCE, [Turn(0.0, 10.0, 0), Turn(10.0, 20.0, 1)], 20.0)

    assert result["der"]["collar_0.0"]["der"] == 0.0
    assert result["purity"] == 1.0
    assert result["coverage"] == 1.0
    assert (result["reference_speakers"], result["speakers_found"]) == (2, 2)


def test_label_names_do_not_matter() -> None:
    result = benchmark.score_turns(REFERENCE, [Turn(0.0, 10.0, 7), Turn(10.0, 20.0, 3)], 20.0)

    assert result["der"]["collar_0.0"]["der"] == 0.0


def test_split_speaker_is_pure_but_not_covered() -> None:
    turns = [Turn(0.0, 10.0, 0), Turn(10.0, 15.0, 1), Turn(15.0, 20.0, 2)]

    result = benchmark.score_turns(REFERENCE, turns, 20.0)

    assert result["purity"] == 1.0
    assert result["coverage"] == 0.75
    assert result["der"]["collar_0.0"]["confusion_seconds"] == 5.0


def test_missing_turn_counts_as_missed_speech() -> None:
    result = benchmark.score_turns(REFERENCE, [Turn(0.0, 10.0, 0)], 20.0)

    assert result["der"]["collar_0.0"]["missed_seconds"] == 10.0
    assert result["der"]["collar_0.0"]["der"] == 0.5


def test_collar_forgives_boundary_errors() -> None:
    turns = [Turn(0.0, 10.2, 0), Turn(10.2, 20.0, 1)]

    result = benchmark.score_turns(REFERENCE, turns, 20.0)

    assert result["der"]["collar_0.0"]["der"] > 0
    assert result["der"]["collar_0.5"]["der"] == 0.0


def test_rttm_round_trip(tmp_path: Path) -> None:
    turns = [simulate.PlannedTurn(0, "1131", 0.5, 0.0, 12.25), simulate.PlannedTurn(3, "428", 13.0, 2.0, 1.5)]
    path = tmp_path / "ref.rttm"
    path.write_text("\n".join(simulate.rttm_lines(turns, "conv")) + "\n", encoding="utf-8")

    assert benchmark.read_rttm(path) == [(0.5, 12.75, "1131"), (13.0, 14.5, "428")]


def utterances(speakers: int, per_speaker: int) -> list[object]:
    return [
        simulate.Utterance(row, f"s{row % speakers}", "main" if row % 3 else "other", 10.0 + row % 7)
        for row in range(speakers * per_speaker)
    ]


def test_schedule_is_reproducible_and_never_repeats_a_speaker() -> None:
    plan = simulate.plan_turns(utterances(5, 40), speakers=3, total_seconds=300, seed=4)

    assert plan == simulate.plan_turns(utterances(5, 40), speakers=3, total_seconds=300, seed=4)
    assert len({turn.speaker for turn in plan}) == 3
    assert all(a.speaker != b.speaker for a, b in itertools.pairwise(plan))


def test_schedule_has_gaps_short_turns_and_one_session_per_speaker() -> None:
    items = utterances(4, 60)
    plan = simulate.plan_turns(items, speakers=4, total_seconds=600, seed=1)

    gaps = [b.start - a.end for a, b in itertools.pairwise(plan)]
    assert all(0.2 - 1e-3 <= gap <= 1.5 + 1e-3 for gap in gaps)
    assert any(turn.seconds <= 3.0 for turn in plan)
    assert all(turn.offset >= 0 and turn.offset + turn.seconds <= items[turn.row].seconds + 1e-3 for turn in plan)
    assert {items[turn.row].session for turn in plan} == {"main"}
    assert plan[-1].end >= 600


def test_schedule_rejects_too_few_speakers() -> None:
    with pytest.raises(simulate.click.UsageError):
        simulate.plan_turns(utterances(2, 10), speakers=3, total_seconds=60, seed=0)


def test_model_hashes_name_candidates_outside_the_models_root(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"weights")
    settings = Settings(models_root=tmp_path / "models", embedding_model=str(candidate))

    hashes = benchmark.model_hashes(settings, benchmark.diarization_paths(settings))

    assert list(hashes) == [  # no full paths
        "pyannote-segmentation-3.0/model.onnx",
        "candidate.onnx",
        "speaker-embedding/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    ]
    assert hashes["candidate.onnx"] == sha256_file(candidate)


def test_model_hashes_hash_candidates_inside_the_models_root(tmp_path: Path) -> None:
    candidate = tmp_path / "speaker-embedding" / "candidate.onnx"
    candidate.parent.mkdir()
    candidate.write_bytes(b"weights")
    settings = Settings(models_root=tmp_path, embedding_model="speaker-embedding/candidate.onnx")

    hashes = benchmark.model_hashes(settings, benchmark.diarization_paths(settings))

    assert hashes["speaker-embedding/candidate.onnx"] == sha256_file(candidate)


def test_rttm_with_several_recordings_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ref.rttm"
    path.write_text(
        "SPEAKER a 1 0.0 1.0 <NA> <NA> x <NA> <NA>\nSPEAKER b 1 0.0 1.0 <NA> <NA> y <NA> <NA>\n", encoding="utf-8"
    )

    with pytest.raises(benchmark.click.UsageError, match="2 recordings"):
        benchmark.read_rttm(path)


def test_optional_string_overrides_accept_none() -> None:
    assert benchmark.apply_overrides(["whisper_model_en=none"]).whisper_model_en is None
    assert benchmark.apply_overrides(["language=it"]).language == "it"
    assert benchmark.apply_overrides([]).whisper_model_en == Settings().whisper_model_en


def test_overrides_convert_each_field_type() -> None:
    settings = benchmark.apply_overrides(
        ["cpu_threads=2", "clustering_threshold=0.85", "vad_filter=no", "models_root=~/models"]
    )
    assert settings.cpu_threads == 2
    assert settings.clustering_threshold == 0.85
    assert settings.vad_filter is False
    assert settings.models_root == Path.home() / "models"
    assert benchmark.apply_overrides(["vad_filter=TRUE"]).vad_filter is True


@pytest.mark.parametrize(
    "item",
    ["no_such_field=1", "clustering_threshold", "clustering_threshold=abc", "cpu_threads=2.5", "vad_filter=flase"],
)
def test_malformed_overrides_are_usage_errors(item: str) -> None:
    with pytest.raises(benchmark.click.BadParameter):
        benchmark.apply_overrides([item])


def test_one_to_one_map_ignores_label_names() -> None:
    counts = {(7, "b"): 5, (3, "a"): 4}
    assert benchmark.best_one_to_one(counts, [3, 7], ["a", "b"]) == (9, {3: "a", 7: "b"})


def test_one_to_one_map_scores_only_one_part_of_a_split_speaker() -> None:
    counts = {(0, "a"): 6, (1, "a"): 4, (2, "b"): 5}
    exact, mapping = benchmark.best_one_to_one(counts, [0, 1, 2], ["a", "b"])
    assert exact == 11
    assert mapping == {0: "a", 2: "b"}


def test_one_to_one_map_scores_only_one_speaker_of_a_merged_cluster() -> None:
    counts = {(0, "a"): 6, (0, "b"): 5}
    assert benchmark.best_one_to_one(counts, [0], ["a", "b"]) == (6, {0: "a"})


def test_one_to_one_map_prefers_the_global_optimum_over_greedy_choices() -> None:
    # Greedy would map cluster 0 to "a" (5) and leave cluster 1 with "b" (1): 6 instead of 8.
    counts = {(0, "a"): 5, (0, "b"): 4, (1, "a"): 4, (1, "b"): 1}
    assert benchmark.best_one_to_one(counts, [0, 1], ["a", "b"])[0] == 8


def test_turn_length_breakdown_counts_unknown_words_as_wrong() -> None:
    triples = [(0, "a", 1), (None, "a", 2), (1, "b", 2), (0, "a", 5), (1, "a", 5)]
    breakdown = benchmark.turn_length_breakdown(triples, {0: "a", 1: "b"}, {0: "a", 1: "a"})
    assert breakdown["turns_1_2_words"] == {
        "one_to_one": round(2 / 3, 4),
        "many_to_one": round(1 / 3, 4),
        "scored_words": 3,
        "unknown_words": 1,
    }
    assert breakdown["turns_3_plus_words"]["one_to_one"] == 0.5
    assert breakdown["turns_3_plus_words"]["many_to_one"] == 1.0


def test_tokens_keep_accented_letters_and_elisions() -> None:
    assert benchmark.tokens("Perché l'uomo è qui — città, ORA!") == ["perché", "l'uomo", "è", "qui", "città", "ora"]
    assert benchmark.tokens("Café snake_case") == ["café", "snake", "case"]  # NFC; "_" splits words
    assert benchmark.tokens("हिन्दी भाषा, ठीक है।") == ["हिन्दी", "भाषा", "ठीक", "है"]  # combining marks stay


def test_reference_lines_give_one_turn_per_line() -> None:
    turns = [simulate.PlannedTurn(1, "1131", 0.5, 0.0, 12.0), simulate.PlannedTurn(0, "428", 13.0, 0.0, 4.0)]

    lines = simulate.reference_lines(turns, ["prima frase", "seconda\n  frase  lunga"])

    assert lines == ["1131\tseconda frase lunga", "428\tprima frase"]


def test_schedule_without_short_share_keeps_whole_utterances() -> None:
    items = [simulate.Utterance(row, f"s{row % 3}", "main", 5.0 + row % 4) for row in range(60)]

    plan = simulate.plan_turns(items, speakers=3, total_seconds=200, seed=2, short_share=0.0)

    assert all(turn.offset == 0.0 and turn.seconds == items[turn.row].seconds for turn in plan)


def test_simulator_refuses_existing_outputs_before_writing(tmp_path: Path) -> None:
    from click.testing import CliRunner

    split = tmp_path / "split.parquet"
    split.write_bytes(b"")
    existing = tmp_path / "talk.rttm"
    existing.write_text("keep", encoding="utf-8")
    args = [str(split), "--seed", "1", "--out", str(tmp_path / "talk.wav"), "--rttm", str(existing)]

    result = CliRunner().invoke(simulate.main, args)

    assert result.exit_code == 2
    assert "refusing to replace existing files" in result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["split.parquet", "talk.rttm"]
    assert existing.read_text(encoding="utf-8") == "keep"


def test_reference_needs_a_transcript_column(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    split = tmp_path / "split.parquet"
    pq.write_table(pa.table({"speaker_id": ["1"]}), split)

    with pytest.raises(simulate.click.UsageError, match="no transcript column"):
        simulate.read_transcripts(split)


sys.modules.setdefault("benchmark", benchmark)  # diarization_lab imports the benchmark helpers by name
lab = load_script("diarization_lab")


def _voices(*means: list[float], noise: float = 0.0, per_voice: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    rows = [np.array(mean) + noise * rng.standard_normal(len(mean)) for mean in means for _ in range(per_voice)]
    vectors: np.ndarray = lab.normalize(np.array(rows, dtype=np.float64))
    return vectors


def _turns(speakers: list[int], seconds: float = 3.0) -> list[Turn]:
    return [Turn(index * 5.0, index * 5.0 + seconds, speaker) for index, speaker in enumerate(speakers)]


def test_average_linkage_merges_close_vectors_only() -> None:
    vectors = _voices([1, 0, 0], [0, 1, 0], noise=0.05)
    assert lab.average_linkage(vectors, np.ones(6), threshold=0.3) == [0, 0, 0, 1, 1, 1]


def test_average_linkage_stops_at_the_oracle_count() -> None:
    vectors = _voices([1, 0, 0], [0, 1, 0], [0, 0, 1], noise=0.05)
    assert len(set(lab.average_linkage(vectors, np.ones(9), threshold=0.0, k=2))) == 2
    assert lab.average_linkage(np.zeros((0, 3)), np.zeros(0), threshold=0.3) == []


def test_recluster_merges_a_split_speaker_and_keeps_close_voices_apart() -> None:
    # sherpa split voice A into clusters 0 and 1; voice B (cluster 2) is close to A but distinct.
    a, b = [1.0, 0.0, 0.0], [0.8, 0.6, 0.0]  # cosine distance 0.2
    vectors = _voices(a, a, b, noise=0.02)
    turns = _turns([0, 0, 0, 1, 1, 1, 2, 2, 2])
    config = lab.Recluster("cluster", threshold=0.1, min_seconds=1.0, reject=0.5)
    labels = [turn.speaker for turn in lab.recluster_turns(turns, vectors, config)]
    assert labels[:6] == [labels[0]] * 6
    assert labels[6:] == [labels[6]] * 3
    assert labels[0] != labels[6]


def test_turn_level_recluster_ignores_sherpa_labels() -> None:
    vectors = _voices([1, 0, 0], [0, 1, 0], noise=0.02)
    turns = _turns([0, 1, 0, 1, 0, 1])  # sherpa mixed the two voices
    config = lab.Recluster("turn", threshold=0.3, min_seconds=1.0, reject=0.5)
    labels = [turn.speaker for turn in lab.recluster_turns(turns, vectors, config)]
    assert labels == [0, 0, 0, 1, 1, 1]


def test_recluster_keeps_a_dissimilar_speaker_heard_only_in_short_turns() -> None:
    vectors = _voices([1, 0, 0], [0, 1, 0], noise=0.02)
    turns = _turns([0, 0, 0]) + [Turn(20.0 + index, 20.5 + index, 1) for index in range(3)]
    config = lab.Recluster("cluster", threshold=0.3, min_seconds=1.0, reject=0.5)
    labels = [turn.speaker for turn in lab.recluster_turns(turns, vectors, config)]
    assert labels[3:] == [labels[3]] * 3
    assert labels[3] != labels[0]


def test_recluster_joins_similar_short_turns_and_keeps_turns_without_embeddings() -> None:
    vectors = _voices([1, 0, 0], noise=0.02, per_voice=4)
    vectors[3] = 0.0  # too short to embed
    turns = [*_turns([0, 0]), Turn(20.0, 20.5, 1), Turn(30.0, 30.2, 2)]
    config = lab.Recluster("cluster", threshold=0.3, min_seconds=1.0, reject=0.5)
    labels = [turn.speaker for turn in lab.recluster_turns(turns, vectors, config)]
    assert labels[:3] == [labels[0]] * 3
    assert labels[3] != labels[0]


def test_recluster_handles_one_speaker_and_no_turns() -> None:
    config = lab.Recluster("cluster", threshold=0.3, min_seconds=1.0, reject=0.5)
    assert {turn.speaker for turn in lab.recluster_turns(_turns([4, 4, 4]), _voices([1, 0], noise=0.02), config)} == {0}
    assert lab.recluster_turns([], np.zeros((0, 2)), config) == []


def test_short_reply_causes_classify_each_wrong_word() -> None:
    words = [Word(float(index), index + 0.4, " w") for index in range(5)]
    turns = [Turn(0.0, 1.0, 0), Turn(1.0, 2.5, 1), Turn(3.0, 3.5, 2), Turn(3.8, 6.0, 0)]  # 3.8: within 0.5 s of word 3
    mapping = {0: "a", 1: "b"}
    scored = [(0, 0, "a"), (1, None, "a"), (2, 2, "a"), (3, 1, "a"), (4, 0, "b")]
    assert lab.short_reply_causes(words, turns, scored, mapping) == {
        "correct": 1,
        "no_turn": 1,
        "unmapped_cluster": 1,
        "near_correct_turn": 1,
        "no_correct_turn_near": 1,
    }


def test_fold_moves_small_similar_groups_into_large_ones_only() -> None:
    vectors = _voices([1, 0, 0], [0.95, 0.31, 0], [0, 1, 0], per_voice=1)
    durations = np.array([30.0, 2.0, 2.0])
    # group 1 is small and similar to group 0; group 2 is small and dissimilar.
    assert lab.fold_small_groups([0, 1, 2], vectors, durations, 10.0, 0.5) == [0, 0, 2]
    assert lab.fold_small_groups([0, 1, 2], vectors, durations, 1.0, 0.5) == [0, 1, 2]  # nothing is small


def test_recluster_name_marks_folding() -> None:
    assert lab.Recluster("turn", 0.4, 2.5, 0.6).name == "turn-t0.4-m2.5-r0.6-kauto"
    assert lab.Recluster("turn", 0.4, 2.5, 0.6, fold_seconds=20, fold_similarity=0.3).name.endswith("-f20s0.3")
