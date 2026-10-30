"""The pipeline steps, in run order; each takes and returns `transcribe.domain` records or arrays.

1. `decode`: audio file -> 16 kHz mono float32 array (PyAV).
2. `language`: array -> `Language` (faster-whisper), unless `--language` forces one.
3. `asr`: array -> timed `Word`s (faster-whisper on the CPU, or mlx-whisper on the Apple GPU).
4. `diarization`: array -> speaker `Turn`s (sherpa-onnx), re-clustered from per-turn embeddings.
5. `attribution`: words and turns -> `AttributedWord`s and `Paragraph`s (pure).
6. `rendering`: paragraphs -> Markdown document (pure).
7. `output`: document -> stdout or a new file, never overwritten.
"""
