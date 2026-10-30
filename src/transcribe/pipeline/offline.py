"""Keep the engine packages off the network."""

import os


def offline_imports() -> None:
    """Forbid Hugging Face Hub network access before faster-whisper is imported."""
    os.environ["HF_HUB_OFFLINE"] = "1"
