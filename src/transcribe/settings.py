"""All tunables in one place; the CLI always uses the defaults."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS_ROOT = Path.home() / "Library" / "Application Support" / "transcribe" / "models"


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Pipeline configuration. Benchmarks and tests override fields; the CLI does not.

    Models:
        models_root: Directory holding the provisioned models
            (`scripts/provision_models.py`). The CLI reads models only from here.
        verify_model_hashes: Check each model file's sha256 against
            `model_manifest.json` before loading it. Off by default, so a start skips
            hashing about 510 MB; provisioning always verifies.

    Speech recognition (faster-whisper):
        whisper_model: Model directory under `models_root`.
        compute_type: CTranslate2 compute type. `int8` quantizes the weights, which
            lowers memory use and speeds up the CPU; some operations stay in floating
            point.
        cpu_threads: CPU threads for recognition.
        beam_size: Candidate transcripts kept at each decoding step. Larger values
            explore more alternatives and run slower; 1 is greedy decoding.
        vad_filter: Run Silero VAD first and decode only the detected speech, so
            silence is skipped; word times still refer to the original recording.
        condition_on_previous_text: Feed each 30 s window's text to the next window
            as a prompt. True keeps wording more consistent across windows but makes
            recognition more than twice as slow (measured in section 9 of the plan);
            False makes repetition loops less likely.
        asr_batch_size: 0 decodes the windows one after another; above 0, faster-
            whisper's `BatchedInferencePipeline` decodes that many windows at once.

    Speaker diarization (sherpa-onnx):
        segmentation_model, embedding_model: Model files under `models_root`, for
            finding speaker turns and for telling voices apart.
        diarization_threads: CPU threads for both diarization models.
        clustering_threshold: Distance below which two voice clusters merge. Higher
            values give fewer speakers (one voice is split less often, but similar
            voices merge); lower values give more speakers.
        min_duration_on: Speech segments shorter than this many seconds are dropped.
        min_duration_off: Gaps shorter than this many seconds between two segments
            of the same speaker are closed, joining the segments.
        window_shift_ratio: Step of the 10 s segmentation window as a fraction of
            its length (0.1 is a 1 s step). Smaller steps cost more time.

    Audio and transcript layout:
        sample_rate: Rate the audio is resampled to; both engines expect 16 kHz.
        min_audio_seconds: Shorter audio skips both engines and yields an empty
            transcript.
        paragraph_gap_seconds: A pause longer than this starts a new paragraph.
        paragraph_max_seconds: A paragraph that would grow longer than this is split.
    """

    models_root: Path = DEFAULT_MODELS_ROOT
    verify_model_hashes: bool = False

    # Speech recognition (faster-whisper).
    whisper_model: str = "faster-whisper-small.en"
    compute_type: str = "int8"
    cpu_threads: int = 4
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    asr_batch_size: int = 0  # 0: sequential WhisperModel; >0: BatchedInferencePipeline

    # Speaker diarization (sherpa-onnx), starting from the official example.
    segmentation_model: str = "pyannote-segmentation-3.0/model.onnx"
    embedding_model: str = "speaker-embedding/nemo_en_titanet_small.onnx"
    diarization_threads: int = 4
    clustering_threshold: float = 0.8  # tuned on held-out audio (M5); the example uses 0.5
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
