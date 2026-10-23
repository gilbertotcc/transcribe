"""All tunables in one place; the CLI always uses the defaults."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS_ROOT = Path.home() / "Library" / "Application Support" / "transcribe" / "models"


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Pipeline configuration. Benchmarks and tests override fields; the CLI does not."""

    models_root: Path = DEFAULT_MODELS_ROOT
    verify_model_hashes: bool = False

    # Speech recognition (faster-whisper).
    whisper_model: str = "faster-whisper-small.en"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 5
    vad_filter: bool = True
    asr_batch_size: int = 0  # 0: sequential WhisperModel; >0: BatchedInferencePipeline

    # Speaker diarization (sherpa-onnx), starting from the official example.
    segmentation_model: str = "pyannote-segmentation-3.0/model.onnx"
    embedding_model: str = "speaker-embedding/nemo_en_titanet_small.onnx"
    diarization_threads: int = 4
    clustering_threshold: float = 0.5
    min_duration_on: float = 0.3
    min_duration_off: float = 0.5
    window_shift_ratio: float = 0.1

    # Audio and transcript layout.
    sample_rate: int = 16_000
    min_audio_seconds: float = 0.5
    paragraph_gap_seconds: float = 2.0
    paragraph_max_seconds: float = 30.0

    @property
    def whisper_model_dir(self) -> Path:
        return self.models_root / self.whisper_model

    @property
    def segmentation_model_path(self) -> Path:
        return self.models_root / self.segmentation_model

    @property
    def embedding_model_path(self) -> Path:
        return self.models_root / self.embedding_model
