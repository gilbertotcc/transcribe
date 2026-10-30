"""mlx-whisper adapter: audio array to timed words on the Apple GPU (Apple Silicon only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transcribe.domain import Word, clamp_interval
from transcribe.pipeline.offline import offline_imports


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from transcribe.config import Settings
    from transcribe.pipeline.events import ProgressCallback

# MLX keeps freed GPU buffers for reuse; uncapped, that cache grew to about 11 GB on a 60 min recording.
CACHE_LIMIT_BYTES = 512 * 2**20
# Speech is decoded in blocks of about this many seconds, cut between VAD chunks, so progress moves.
BLOCK_SECONDS = 300.0


def group_chunks(chunks: Sequence[dict[str, int]], max_samples: int) -> list[list[dict[str, int]]]:
    """Consecutive VAD chunks grouped so that each group holds at most `max_samples` of speech (or one chunk)."""
    groups: list[list[dict[str, int]]] = []
    size = 0
    for chunk in chunks:
        length = chunk["end"] - chunk["start"]
        if not groups or size + length > max_samples:
            groups.append([])
            size = 0
        groups[-1].append(chunk)
        size += length
    return groups


def _blocks(audio: npt.NDArray[np.float32], settings: Settings) -> list[tuple[npt.NDArray[np.float32], Any, float]]:
    """(speech, time map to the original recording or None, end in the original) for each block."""
    if not settings.vad_filter:
        return [(audio, None, len(audio) / settings.sample_rate)]
    import numpy as np
    from faster_whisper.vad import SpeechTimestampsMap, get_speech_timestamps

    rate = settings.sample_rate
    # Grouped here: faster-whisper's collect_chunks(max_duration=...) drops each later block's first chunk
    # from its metadata, which would shift the restored times.
    return [
        (
            np.concatenate([audio[chunk["start"] : chunk["end"]] for chunk in group]),
            SpeechTimestampsMap(group, rate),
            group[-1]["end"] / rate,
        )
        for group in group_chunks(get_speech_timestamps(audio), round(BLOCK_SECONDS * rate))
    ]


def transcribe(
    audio: npt.NDArray[np.float32], settings: Settings, language: str, *, progress: ProgressCallback | None = None
) -> list[Word]:
    offline_imports()
    import mlx.core as mx
    import mlx_whisper  # engines load lazily

    mx.set_cache_limit(CACHE_LIMIT_BYTES)

    # A directory path (never a hub name) is loaded directly; mlx-whisper downloads only for a missing path.
    model_dir = settings.whisper_model_dir_for(language)
    if not model_dir.is_dir():
        msg = f"MLX model directory is missing: {model_dir.name}"
        raise FileNotFoundError(msg)
    duration = len(audio) / settings.sample_rate
    words: list[Word] = []
    for speech, time_map, block_end in _blocks(audio, settings):
        result = mlx_whisper.transcribe(
            speech,
            path_or_hf_repo=str(model_dir),
            language=language,
            word_timestamps=True,
            condition_on_previous_text=settings.condition_on_previous_text,
            verbose=None,  # no output on stdout
        )
        for segment in result["segments"]:
            for word in segment.get("words", ()):
                start, end = float(word["start"]), float(word["end"])
                if time_map is not None:  # start and end resolve to the same chunk, as in faster-whisper
                    chunk = time_map.get_chunk_index((start + end) / 2)
                    start, end = time_map.get_original_time(start, chunk), time_map.get_original_time(end, chunk)
                start, end = clamp_interval(start, end, duration)
                words.append(Word(start, end, word["word"]))
        if progress is not None:
            progress(block_end, duration)
    if progress is not None:
        progress(duration, duration)  # trailing silence yields no block
    _release_gpu_memory()
    return words


def _release_gpu_memory() -> None:
    """Drop the cached model and MLX's buffer cache, so diarization does not run on top of them."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = None
    ModelHolder.model_path = None
    mx.clear_cache()
