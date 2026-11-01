"""Developer tool: summarize Claude Code token usage for this project from its local session logs.

Prints counts and API list-price estimates only, never conversation content, and makes no network calls.
Use `--since` to compare the sessions after a change with the earlier ones:

    uv run scripts/token_report.py --since 2026-11-02
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click


LONG_SESSION_REQUESTS = 100
COLD_RESTART_GAP = timedelta(hours=1)
COLD_RESTART_MIN_WRITE = 30_000


@dataclass(frozen=True)
class Price:
    """USD per million tokens, from https://platform.claude.com/docs/en/about-claude/pricing (2026-11-01)."""

    input: float
    write_5m: float
    write_1h: float
    cache_read: float
    output: float


PRICES = {
    "claude-opus-5-5": Price(input=4, write_5m=5, write_1h=8, cache_read=0.20, output=20),
    "claude-sonnet-5-5": Price(input=2, write_5m=2.5, write_1h=4, cache_read=0.10, output=10),
}


@dataclass(frozen=True)
class Request:
    """Token counts of one API request."""

    session: str
    subagent: bool
    timestamp: datetime
    model: str
    input_tokens: int
    cache_read: int
    write_5m: int
    write_1h: int
    output: int

    @property
    def context(self) -> int:
        return self.input_tokens + self.cache_read + self.write_5m + self.write_1h

    @property
    def total(self) -> int:
        return self.context + self.output


@dataclass(frozen=True)
class Report:
    sessions: int
    requests: int
    total_tokens: int
    cache_read_tokens: int
    average_context: int
    peak_context: int
    long_session_read_share: float
    cold_restarts: int
    cold_restart_tokens: int
    read_cost: float
    write_cost: float
    output_cost: float
    unpriced_tokens: int

    @property
    def priced_cost(self) -> float:
        return self.read_cost + self.write_cost + self.output_cost


def project_log_dir(project: Path, home: Path) -> Path:
    """Return the directory where Claude Code keeps the session logs of `project`."""
    return home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(project))


def _request(record: dict[str, Any], session: str, *, subagent: bool) -> Request:
    message = record["message"]
    usage = message["usage"]
    written = usage.get("cache_creation_input_tokens", 0)
    write_1h = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0)
    return Request(
        session=session,
        subagent=subagent,
        timestamp=datetime.fromisoformat(record["timestamp"]),
        model=message.get("model", "unknown"),
        input_tokens=usage.get("input_tokens", 0),
        cache_read=usage.get("cache_read_input_tokens", 0),
        write_5m=written - write_1h,
        write_1h=write_1h,
        output=usage.get("output_tokens", 0),
    )


def read_requests(log_dir: Path) -> list[Request]:
    """Read the requests of every session log, counting a request once however many lines repeat it."""
    requests = []
    for path in sorted(log_dir.rglob("*.jsonl")):
        seen: set[str] = set()
        subagent = "subagents" in path.parts
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("type") != "assistant" or "timestamp" not in record:
                continue
            message = record.get("message")
            if not isinstance(message, dict) or not message.get("usage"):
                continue
            if message.get("model") == "<synthetic>" or message.get("id") in seen:
                continue
            seen.add(message.get("id", ""))
            requests.append(_request(record, path.stem, subagent=subagent))
    return requests


def summarize(requests: list[Request]) -> Report:
    """Aggregate requests; cold restarts and session lengths look at main conversations only."""
    by_session: dict[str, list[Request]] = {}
    for request in requests:
        if not request.subagent:
            by_session.setdefault(request.session, []).append(request)

    cold_restarts = cold_restart_tokens = 0
    long_session_reads = 0
    for session in by_session.values():
        session.sort(key=lambda request: request.timestamp)
        if len(session) >= LONG_SESSION_REQUESTS:
            long_session_reads += sum(request.cache_read for request in session)
        for previous, current in itertools.pairwise(session):
            written = current.write_5m + current.write_1h
            if written >= COLD_RESTART_MIN_WRITE and current.timestamp - previous.timestamp > COLD_RESTART_GAP:
                cold_restarts += 1
                cold_restart_tokens += written

    read_cost = write_cost = output_cost = 0.0
    unpriced = 0
    for request in requests:
        price = PRICES.get(request.model)
        if price is None:
            unpriced += request.total
            continue
        read_cost += request.cache_read * price.cache_read / 1e6
        write_cost += (request.write_5m * price.write_5m + request.write_1h * price.write_1h) / 1e6
        output_cost += (request.output * price.output + request.input_tokens * price.input) / 1e6

    main_contexts = [request.context for session in by_session.values() for request in session]
    cache_read = sum(request.cache_read for request in requests)
    main_reads = sum(request.cache_read for session in by_session.values() for request in session)
    return Report(
        sessions=len(by_session),
        requests=len(requests),
        total_tokens=sum(request.total for request in requests),
        cache_read_tokens=cache_read,
        average_context=sum(main_contexts) // len(main_contexts) if main_contexts else 0,
        peak_context=max(main_contexts, default=0),
        long_session_read_share=long_session_reads / main_reads if main_reads else 0.0,
        cold_restarts=cold_restarts,
        cold_restart_tokens=cold_restart_tokens,
        read_cost=read_cost,
        write_cost=write_cost,
        output_cost=output_cost,
        unpriced_tokens=unpriced,
    )


def format_report(report: Report) -> str:
    """Render the report as plain text lines."""
    per_request = report.total_tokens // report.requests if report.requests else 0
    lines = [
        f"Sessions: {report.sessions}, requests: {report.requests}",
        f"Total tokens: {report.total_tokens:,} ({per_request:,} per request)",
        f"Cache reads: {report.cache_read_tokens:,}",
        f"Context per main request: average {report.average_context:,}, peak {report.peak_context:,}",
        f"Cache reads in sessions of {LONG_SESSION_REQUESTS}+ requests: {report.long_session_read_share:.0%}",
        f"Cold restarts: {report.cold_restarts} ({report.cold_restart_tokens:,} tokens rewritten)",
    ]
    if report.priced_cost:
        lines += [
            f"List-price estimate: ${report.priced_cost:,.2f}",
            f"  cache reads {report.read_cost / report.priced_cost:.0%}, "
            f"cache writes {report.write_cost / report.priced_cost:.0%}, "
            f"input and output {report.output_cost / report.priced_cost:.0%}",
        ]
    if report.unpriced_tokens:
        lines.append(f"Tokens on models without a price here: {report.unpriced_tokens:,}")
    return "\n".join(lines)


@click.command()
@click.option("--project", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path.cwd)
@click.option("--since", type=click.DateTime(formats=["%Y-%m-%d"]), help="Count only requests from this UTC date on.")
def main(project: Path, since: datetime | None) -> None:
    """Summarize Claude Code token usage for PROJECT (default: the current directory)."""
    requests = read_requests(project_log_dir(project.resolve(), Path.home()))
    if since is not None:
        start = since.replace(tzinfo=UTC)
        requests = [request for request in requests if request.timestamp >= start]
    click.echo(format_report(summarize(requests)))


if __name__ == "__main__":
    main()
