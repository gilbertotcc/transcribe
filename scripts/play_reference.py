"""Developer tool: build a speaker-tagged benchmark reference from a Project Gutenberg play.

Each output line is `SPEAKER<TAB>text`; `-` marks text with no known reader
(stage directions, scene descriptions), which counts for WER but not for speaker
attribution. Character names are mapped to readers with repeated `--cast`
options, because one reader may voice several characters:

    uv run scripts/play_reference.py pg844.txt --start "FIRST ACT" --end "THIRD ACT" \\
        --cast LANE=toby --cast MERRIMAN=toby --cast ALGERNON=simon ... > reference.tsv
"""

from __future__ import annotations

import re
from pathlib import Path

import click


UNSCORED = "-"
_CHARACTER_LINE = re.compile("^([A-Z][A-Z .\u2019']+)\\.$")
_STAGE_DIRECTION = re.compile(r"\[[^\]]*\]")
_MARKUP = re.compile(r"_")


def parse_play(text: str, cast: dict[str, str]) -> list[tuple[str, str]]:
    """Split a play into (speaker, text) chunks in reading order."""
    chunks: list[tuple[str, str]] = []
    speaker = UNSCORED
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().splitlines()
        if not lines:
            continue
        match = _CHARACTER_LINE.match(lines[0].strip())
        if match and match.group(1).strip() in cast:
            speaker = cast[match.group(1).strip()]
            body = " ".join(lines[1:])
        else:
            speaker = UNSCORED  # scene descriptions, act headings
            body = " ".join(lines)
        body = _MARKUP.sub("", body)
        position = 0
        for direction in _STAGE_DIRECTION.finditer(body):
            chunks.append((speaker, body[position : direction.start()]))
            chunks.append((UNSCORED, direction.group(0)[1:-1]))
            position = direction.end()
        chunks.append((speaker, body[position:]))
    return [(who, " ".join(words.split())) for who, words in chunks if words.strip()]


@click.command()
@click.argument("play", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--start", required=True, help="Line where the reference starts (inclusive).")
@click.option("--end", required=True, help="Line where the reference stops (exclusive).")
@click.option("--cast", "cast_items", multiple=True, required=True, metavar="CHARACTER=READER")
def main(play: Path, start: str, end: str, cast_items: tuple[str, ...]) -> None:
    """Print a speaker-tagged reference for the part of PLAY between --start and --end."""
    cast = dict(item.split("=", 1) for item in cast_items)
    text = play.read_text(encoding="utf-8")
    begin = text.index(f"\n{start}\n")
    stop = text.index(f"\n{end}\n", begin)
    for speaker, words in parse_play(text[begin:stop], cast):
        click.echo(f"{speaker}\t{words}")


if __name__ == "__main__":
    main()
