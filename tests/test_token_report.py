"""scripts/token_report.py: token counts from synthetic session logs, never their content."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner


if TYPE_CHECKING:
    from types import ModuleType


PRIVATE_TEXT = "recognized text that must stay private"


def load_token_report() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "token_report.py"
    spec = importlib.util.spec_from_file_location("scripts_token_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module while the class is created
    spec.loader.exec_module(module)
    return module


token_report = load_token_report()


def assistant(
    message_id: str,
    timestamp: str,
    *,
    read: int = 0,
    write_1h: int = 0,
    output: int = 10,
    model: str = "claude-opus-5-5",
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "id": message_id,
            "model": model,
            "content": [{"type": "text", "text": PRIVATE_TEXT}],
            "usage": {
                "input_tokens": 1,
                "output_tokens": output,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write_1h,
                "cache_creation": {"ephemeral_1h_input_tokens": write_1h},
            },
        },
    }


def write_log(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join([*lines, "not json"]), encoding="utf-8")


def test_project_log_dir_replaces_non_alphanumerics() -> None:
    log_dir = token_report.project_log_dir(Path("/Users/me/my_repo.x"), Path("/Users/me"))
    assert log_dir == Path("/Users/me/.claude/projects/-Users-me-my-repo-x")


def test_read_requests_counts_repeated_lines_once_and_skips_synthetic(tmp_path: Path) -> None:
    first = assistant("a", "2026-11-01T10:00:00+00:00", read=100)
    synthetic = assistant("b", "2026-11-01T10:01:00+00:00", model="<synthetic>")
    write_log(tmp_path / "s.jsonl", [first, first, synthetic, {"type": "user", "message": {"content": PRIVATE_TEXT}}])

    requests = token_report.read_requests(tmp_path)

    assert [request.cache_read for request in requests] == [100]
    assert requests[0].context == 101
    assert requests[0].total == 111


def test_summarize_finds_cold_restarts_and_long_sessions(tmp_path: Path) -> None:
    records = [assistant(f"m{i}", f"2026-11-01T10:{i % 60:02d}:00+00:00", read=1000) for i in range(100)]
    records.append(assistant("late", "2026-11-01T13:00:00+00:00", write_1h=40_000))
    write_log(tmp_path / "long.jsonl", records)
    write_log(tmp_path / "short.jsonl", [assistant("x", "2026-11-01T10:00:00+00:00", read=100_000)])

    report = token_report.summarize(token_report.read_requests(tmp_path))

    assert report.sessions == 2
    assert report.requests == 102
    assert report.cold_restarts == 1
    assert report.cold_restart_tokens == 40_000
    assert report.long_session_read_share == pytest.approx(0.5)
    assert report.peak_context == 100_001


def test_summarize_prices_known_models_and_flags_others(tmp_path: Path) -> None:
    write_log(
        tmp_path / "s.jsonl",
        [
            assistant("a", "2026-11-01T10:00:00+00:00", read=1_000_000, write_1h=1_000_000, output=0),
            assistant("b", "2026-11-01T10:01:00+00:00", output=50, model="claude-other"),
        ],
    )

    report = token_report.summarize(token_report.read_requests(tmp_path))

    assert report.read_cost == pytest.approx(0.20)
    assert report.write_cost == pytest.approx(8.0)
    assert report.unpriced_tokens == 51


def test_summarize_of_no_requests_is_empty() -> None:
    report = token_report.summarize([])
    assert report.requests == 0
    assert "Sessions: 0" in token_report.format_report(report)


def test_cli_filters_by_date_and_prints_no_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    log_dir = token_report.project_log_dir(project.resolve(), tmp_path)
    write_log(
        log_dir / "s.jsonl",
        [
            assistant("old", "2026-10-01T10:00:00+00:00"),
            assistant("new", "2026-11-02T10:00:00+00:00"),
        ],
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(token_report.main, ["--project", str(project), "--since", "2026-11-02"])

    assert result.exit_code == 0
    assert "requests: 1" in result.output
    assert PRIVATE_TEXT not in result.output
