"""Developer tool: build a conversation with known speaker turns from one MLS Parquet split.

    uv run scripts/simulate_conversation.py MLS.parquet --speakers 6 --minutes 20 --seed 1 \\
        --out bench/clips/sim_it_s1.wav --rttm bench/clips/sim_it_s1.rttm [--reference sim_it_s1.tsv]

MLS is Multilingual LibriSpeech (one reader per utterance); download a split by
hand into `bench/` (see docs/benchmark.md). Each turn is one utterance of a
randomly chosen speaker, never the previous one; some turns are cropped to a
few seconds so short turns occur. Turns never overlap. Every utterance is
scaled to the same RMS level, so loudness does not tell speakers apart. The
RTTM holds times and MLS speaker IDs only, no text.

`--reference` also writes the MLS transcript of each turn, one `speaker<TAB>text`
line per turn, the format `scripts/benchmark.py --reference` reads. The text of a
cropped turn is unknown, so with `--reference` no turn is cropped. The file holds
text: keep it in `bench/`, like the audio.
"""

from __future__ import annotations

import io
import random
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np
import pyarrow.parquet as pq

from transcribe.audio import offline_imports


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

SAMPLE_RATE = 16_000
SESSION_COLUMNS = ("chapter_id", "book_id")  # Italian MLS names sessions by chapter, English MLS by book
TARGET_RMS = 0.05  # about -26 dBFS
LEAD_IN_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class Utterance:
    row: int
    speaker: str
    session: str
    seconds: float


@dataclass(frozen=True, slots=True)
class PlannedTurn:
    row: int
    speaker: str
    start: float  # in the conversation
    offset: float  # into the utterance, where a cropped turn begins
    seconds: float

    @property
    def end(self) -> float:
        return self.start + self.seconds


def plan_turns(
    utterances: Sequence[Utterance],
    *,
    speakers: int,
    total_seconds: float,
    seed: int,
    short_share: float = 0.25,
    short_seconds: tuple[float, float] = (1.0, 3.0),
    gap_seconds: tuple[float, float] = (0.2, 1.5),
) -> list[PlannedTurn]:
    """Seeded turn schedule: alternating random speakers, each from one recording session."""
    rng = random.Random(seed)  # noqa: S311 - reproducible simulation, not cryptography
    by_speaker: dict[str, list[Utterance]] = {}
    for item in utterances:
        by_speaker.setdefault(item.speaker, []).append(item)
    if len(by_speaker) < speakers:
        msg = f"the split has {len(by_speaker)} speakers, fewer than {speakers}"
        raise click.UsageError(msg)
    chosen = rng.sample(sorted(by_speaker), speakers)
    pools: dict[str, list[Utterance]] = {}
    for speaker in chosen:
        # The session with the most utterances, so a speaker keeps one microphone and room.
        sessions = Counter(item.session for item in by_speaker[speaker])
        session = min(sessions, key=lambda name: (-sessions[name], name))
        pool = sorted((item for item in by_speaker[speaker] if item.session == session), key=lambda item: item.row)
        rng.shuffle(pool)
        pools[speaker] = pool

    turns: list[PlannedTurn] = []
    cursor = LEAD_IN_SECONDS
    current: str | None = None
    while cursor < total_seconds:
        current = rng.choice([speaker for speaker in chosen if speaker != current])
        if not pools[current]:
            msg = f"speaker {current} ran out of utterances; ask for fewer minutes or more speakers"
            raise click.UsageError(msg)
        utterance = pools[current].pop()
        offset, seconds = 0.0, utterance.seconds
        if rng.random() < short_share:
            seconds = min(rng.uniform(*short_seconds), utterance.seconds)
            offset = rng.uniform(0.0, utterance.seconds - seconds)
        turns.append(PlannedTurn(utterance.row, current, round(cursor, 3), round(offset, 3), round(seconds, 3)))
        cursor = turns[-1].end + rng.uniform(*gap_seconds)
    return turns


def rttm_lines(turns: Sequence[PlannedTurn], uri: str) -> list[str]:
    return [f"SPEAKER {uri} 1 {t.start:.3f} {t.seconds:.3f} <NA> <NA> {t.speaker} <NA> <NA>" for t in turns]


def reference_lines(turns: Sequence[PlannedTurn], transcripts: Sequence[str]) -> list[str]:
    """One `speaker<TAB>text` line per turn; `transcripts` is the split's transcript column."""
    return [f"{turn.speaker}\t{' '.join(transcripts[turn.row].split())}" for turn in turns]


def read_transcripts(path: Path) -> list[str]:
    return [str(text) for text in pq.read_table(path, columns=["transcript"]).column("transcript").to_pylist()]


def read_utterances(path: Path) -> list[Utterance]:
    names = pq.read_schema(path).names
    session = next((name for name in SESSION_COLUMNS if name in names), None)
    if session is None:
        msg = f"no session column ({', '.join(SESSION_COLUMNS)}) in the split"
        raise click.UsageError(msg)
    table = pq.read_table(path, columns=["speaker_id", session, "audio_duration"]).to_pydict()
    return [
        Utterance(row, str(speaker), str(chapter), float(seconds))
        for row, (speaker, chapter, seconds) in enumerate(
            zip(table["speaker_id"], table[session], table["audio_duration"], strict=True)
        )
    ]


def render(path: Path, turns: Sequence[PlannedTurn]) -> npt.NDArray[np.float32]:
    offline_imports()
    from faster_whisper import decode_audio  # PyAV decoder, as in transcribe.audio

    audio_column = pq.read_table(path, columns=["audio"]).column("audio")
    total = round((turns[-1].end + LEAD_IN_SECONDS) * SAMPLE_RATE)
    out = np.zeros(total, dtype=np.float32)
    for turn in turns:
        encoded: bytes = audio_column[turn.row].as_py()["bytes"]
        samples: npt.NDArray[np.float32] = decode_audio(io.BytesIO(encoded), sampling_rate=SAMPLE_RATE)
        begin = round(turn.offset * SAMPLE_RATE)
        piece = samples[begin : begin + round(turn.seconds * SAMPLE_RATE)]
        rms = float(np.sqrt(np.mean(piece**2))) if piece.size else 0.0
        if rms > 0:
            piece = piece * (TARGET_RMS / rms)
        at = round(turn.start * SAMPLE_RATE)
        out[at : at + piece.size] = piece[: total - at]
    return np.clip(out, -1.0, 1.0)


def write_wav(path: Path, samples: npt.NDArray[np.float32]) -> None:
    with path.open("xb") as raw, wave.open(raw, "wb") as stream:  # "x": never replace an existing clip
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes((samples * 32767).astype("<i2").tobytes())


@click.command()
@click.argument("split", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--speakers", type=click.IntRange(min=2), default=6, show_default=True)
@click.option("--minutes", type=click.FloatRange(min=0.5), default=20.0, show_default=True)
@click.option("--seed", type=int, required=True)
@click.option("--out", "out_wav", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--rttm", "out_rttm", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option(
    "--reference",
    "out_reference",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write the transcript reference (TSV); no turn is cropped.",
)
def main(
    split: Path, speakers: int, minutes: float, seed: int, out_wav: Path, out_rttm: Path, out_reference: Path | None
) -> None:
    """Write a simulated conversation WAV, its RTTM reference and, optionally, its transcript reference."""
    turns = plan_turns(
        read_utterances(split),
        speakers=speakers,
        total_seconds=minutes * 60,
        seed=seed,
        short_share=0.0 if out_reference is not None else 0.25,
    )
    write_wav(out_wav, render(split, turns))
    with out_rttm.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(rttm_lines(turns, out_wav.stem)) + "\n")
    if out_reference is not None:
        with out_reference.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(reference_lines(turns, read_transcripts(split))) + "\n")
    click.echo(f"{len(turns)} turns, {speakers} speakers, {turns[-1].end / 60:.1f} min", err=True)


if __name__ == "__main__":
    main()
