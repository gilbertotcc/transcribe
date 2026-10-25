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
            hashing about 2.5 GB; provisioning always verifies.

    Language:
        language: Whisper language code to use instead of detecting one; the CLI
            sets it from `--language`. None detects the language.
        language_detection_seconds: Detection looks for speech only in this opening
            part of the recording, so its VAD pass stays short; the whole recording
            is searched only when the opening holds no speech.
        language_detection_segments: Up to this many 30 s windows of speech are
            tried. Detection stops at the first window whose top language is more
            likely than `language_detection_threshold`; when none is, the windows
            vote and a warning suggests `--language`.
        language_detection_threshold: Probability above which one window decides.

    Speech recognition (faster-whisper):
        whisper_model: Multilingual model directory under `models_root`. It always
            detects the language; without `whisper_model_large` it also transcribes
            every language but English.
        whisper_model_en: English-only model directory, used without
            `whisper_model_large` when the language is English (detected or
            forced). None uses `whisper_model` for English too.
        whisper_model_large: Larger multilingual model that transcribes every
            language. None (the CLI's `--quick`) transcribes with `whisper_model`
            and `whisper_model_en` instead: faster, with lower accuracy.
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
            recognition more than twice as slow (measured in `docs/benchmark.md`);
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

    # Language.
    language: str | None = None
    language_detection_seconds: float = 600.0
    language_detection_segments: int = 3
    language_detection_threshold: float = 0.5

    # Speech recognition (faster-whisper).
    whisper_model: str = "faster-whisper-small"
    whisper_model_en: str | None = "faster-whisper-small.en"
    whisper_model_large: str | None = "faster-whisper-large-v3-turbo"
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
    clustering_threshold: float = 0.8  # tuned on held-out audio (docs/benchmark.md); the example uses 0.5
    min_duration_on: float = 0.3
    min_duration_off: float = 0.5
    window_shift_ratio: float = 0.1

    # Audio and transcript layout.
    sample_rate: int = 16_000
    min_audio_seconds: float = 0.5
    paragraph_gap_seconds: float = 2.0
    paragraph_max_seconds: float = 30.0

    @property
    def whisper_models(self) -> tuple[str, ...]:
        """Every Whisper model a run may load: detection first, unless the language is forced."""
        if self.language is not None:
            return (self.whisper_model_for(self.language),)
        if self.whisper_model_large is not None:
            recognizers: tuple[str, ...] = (self.whisper_model_large,)
        else:
            recognizers = (self.whisper_model_for("en"), self.whisper_model)
        return tuple(dict.fromkeys((self.whisper_model, *recognizers)))

    @property
    def whisper_model_dir(self) -> Path:
        return self.models_root / self.whisper_model

    def whisper_model_for(self, language: str) -> str:
        """Name of the model that transcribes `language`."""
        if self.whisper_model_large is not None:
            return self.whisper_model_large
        if language == "en" and self.whisper_model_en is not None:
            return self.whisper_model_en
        return self.whisper_model

    def whisper_model_dir_for(self, language: str) -> Path:
        """Directory of the model that transcribes `language`."""
        return self.models_root / self.whisper_model_for(language)

    @property
    def segmentation_model_path(self) -> Path:
        return self.models_root / self.segmentation_model

    @property
    def embedding_model_path(self) -> Path:
        return self.models_root / self.embedding_model
