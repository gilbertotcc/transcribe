"""Developer tools in scripts/: time-based diarization scoring and the conversation simulator (no models)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from transcribe.domain import Turn
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

    assert list(hashes) == ["pyannote-segmentation-3.0/model.onnx", "candidate.onnx"]  # no full paths
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
