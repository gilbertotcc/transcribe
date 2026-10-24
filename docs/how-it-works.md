# How transcribe turns audio into a speaker-attributed transcript

This document explains what happens between `transcribe meeting.m4a` and the
Markdown it prints, which scientific work each step relies on, and what the
reported accuracy numbers do and do not measure. It describes the code at the
versions locked in `uv.lock` (`faster-whisper` 1.2.1, `ctranslate2` 4.8.3,
`sherpa-onnx` 2.0.0, `jiwer` 4.0.0). Claude Code wrote this application; it is
not the model that recognizes speech at runtime.

Three kinds of statement appear below, and they are kept apart:

- **Mechanism**: how a component works, with a link to its paper, model card
  or source code.
- **Measured here**: a number from this repository's benchmark on the 2020
  Intel MacBook Pro (section 9 of the
  [specification](../idea/transcribe-mvp-agent-plan.md)).
- **Hypothesis**: a plausible explanation that no experiment has tested yet.

## 1. The pipeline at a glance

```text
                    AUDIOFILE (WAV, MP3, M4A, ...)
                                 |
                  audio.py: decode + resample (PyAV)
                                 |
                 16 kHz mono float32 waveform (numpy)
                                 |
           +---------------------+----------------------+
           | step 1                                     | step 2
           v                                            v
 asr.py: faster-whisper                       diarization.py: sherpa-onnx
   Silero VAD (speech regions)                  pyannote segmentation 3.0
   Whisper small.en (CTranslate2, int8)         TitaNet small embeddings
   word timestamps (attention + DTW)            agglomerative clustering
           |                                            |
           v                                            v
   words with start/end times                  speaker turns with start/end
   (" Hello", 3.20, 3.48)                      (3.10, 5.90, cluster 4)
           |                                            |
           +---------------------+----------------------+
                                 v
               attribution.py: largest time overlap per word
                                 |
               attribution.py: paragraphs by speaker and gaps
                                 |
                   rendering.py: escaped Markdown
                                 |
                  output.py: stdout or a new .md file
```

The two branches read the same waveform but run one after the other, not in
parallel (`application.build_transcript`). Neither branch knows about the
other: the recognizer never sees speaker labels, and the diarizer never sees
words. They meet only through timestamps.

### How speech becomes text, in one paragraph

Whisper does not look up words in the audio, and the audio contains no
embedded text to extract. It is a neural network trained on hundreds of
thousands of hours of audio paired with transcripts, so that given a
spectrogram it **predicts** the most likely text, one token at a time. The
transcript is the model's best guess; it is usually right, but it can be
fluent and wrong, including inventing words that were never spoken. Every
later stage (speaker turns, attribution, Markdown) is deterministic
bookkeeping around that guess, not a further language-model pass.

## 2. Stage by stage

### 2.1 Decoding (`audio.py`)

`decode()` opens the file itself and passes the file object to
`faster_whisper.decode_audio(..., sampling_rate=16000)`. That function uses
PyAV (Python bindings to FFmpeg's libraries) to read the first audio stream,
downmix it to mono and resample it to 16 kHz float32 samples. This is signal
conversion only; nothing is recognized yet. One hour of audio becomes about
230 MB of samples, which both branches share.

### 2.2 Speech recognition (`asr.py`, `settings.py`)

Four different things are easy to conflate here:

| Layer        | What it is in this project                                                 |
| ------------ | -------------------------------------------------------------------------- |
| Architecture | Whisper: Transformer encoder-decoder (Radford et al., reading list item 1) |
| Checkpoint   | `small.en`: an English-only Whisper model released by OpenAI               |
| Conversion   | `Systran/faster-whisper-small.en`: the checkpoint converted to CTranslate2 |
| Runtime      | CTranslate2 4.8.3, called through faster-whisper 1.2.1, on the CPU         |

`asr.transcribe()` builds `WhisperModel(<model directory>, device="cpu",
compute_type="int8", cpu_threads=4, local_files_only=True)` and calls
`model.transcribe(audio, language="en", beam_size=5,
condition_on_previous_text=False, word_timestamps=True, vad_filter=True)`.
With `Settings.asr_batch_size` above 0 it uses faster-whisper's
`BatchedInferencePipeline` instead; the default (0) is sequential.

**Mechanism, per the Whisper paper.** The waveform is cut into 30-second
windows and turned into a log-Mel spectrogram (80 frequency bands for this
checkpoint, as reported by CTranslate2). The encoder turns the spectrogram
into a sequence of hidden vectors; the decoder then generates text tokens
autoregressively, each conditioned on the audio and on the tokens already
produced. Training is supervised next-token prediction on audio paired with
(often imperfect) internet transcripts, which the paper calls weak
supervision. The paper's model family is multilingual and multitask; this
project uses one of its English-only checkpoints with the language fixed to
English, so no language detection takes place.

**Beam search** (`beam_size=5`) keeps the five most likely partial
transcripts at each step instead of only the single best token, and returns
the best complete one. It costs more compute than greedy decoding.

**`condition_on_previous_text=False`.** By default Whisper feeds the previous
window's text into the next window as a prompt. The faster-whisper docstring
states the trade-off: disabling it "may make the text inconsistent across
windows, but the model becomes less prone to getting stuck in a failure
loop, such as repetition looping or timestamps going out of sync". The MVP
disables it; it has not been measured with it enabled.

**INT8.** CTranslate2 quantizes the weights of embedding and linear layers
to 8-bit integers with a per-row scale; all other layers run in floating
point. On this CPU, requesting `int8` gives the effective compute type
`int8_float32` (measured here with `ctranslate2.models.Whisper.compute_type`).
Not all arithmetic is 8-bit, and the speed-up and accuracy effect have not
been measured separately in this project. Quantization is an inference
optimization: the architecture and the recognized vocabulary stay the same.

The returned segments are a lazy generator; inference happens while
`asr.transcribe()` iterates over them. Each faster-whisper word keeps its
leading space (`" Hello"`), and `attribution.paragraphs()` concatenates words
verbatim.

### 2.3 Speech regions and timestamps

Two separate mechanisms produce times, and neither is a phoneme-level forced
aligner.

**Voice activity detection (VAD).** With `vad_filter=True`, faster-whisper
first runs the Silero VAD model (`faster_whisper/assets/silero_vad_v6.onnx`)
to find stretches that contain speech. VAD answers only "is someone speaking
here?", not what was said and not who said it. The sequential path
concatenates the speech regions, transcribes that shorter audio, and then
`restore_speech_timestamps()` maps every word back to the time in the
original recording. The engine test
`test_leading_silence_keeps_original_timestamps` checks this: prepending 10 s
of silence shifts every word by about 10 s. Silero VAD is unrelated to the
pyannote segmentation model used for diarization.

**Word timestamps.** Whisper is trained to emit segment-level timestamps, not
word times. With `word_timestamps=True`, faster-whisper's `find_alignment()`
asks CTranslate2 to align the recognized tokens to the audio frames: it takes
the decoder's cross-attention weights from the alignment heads listed in the
model's `config.json`, smooths them with a median filter, and finds a
monotonic path with dynamic time warping (DTW). Word boundaries are therefore
**estimates** derived from attention, not measurements, and they can be off by
a fraction of a second, especially around pauses.

### 2.4 Who spoke when (`diarization.py`)

Speaker diarization answers "who spoke when" with anonymous labels. It knows
nothing about words or real names. The sherpa-onnx `OfflineSpeakerDiarization`
pipeline used here has three parts, all running through ONNX Runtime:

1. **Local segmentation**: the pyannote segmentation 3.0 model, exported to
   ONNX by sherpa-onnx. Per its model card, it ingests 10 seconds of 16 kHz
   mono audio and outputs, for each frame, one of 7 "powerset" classes: no
   speech, one of three local speakers, or one of the pairs of them speaking
   at once. So within one 10-second window it can describe up to three
   speakers, at most two simultaneously. These are limits per window, not for
   the recording. sherpa-onnx slides the window by
   `window_shift_ratio = 0.1` of its length (1 s) and converts the powerset
   classes back to per-speaker activity. Local speaker 1 in one window has no
   relation to local speaker 1 in the next; the model card states that this
   model "cannot be used to perform speaker diarization of full recordings on
   its own".
2. **Speaker embeddings**: NVIDIA NeMo TitaNet small, exported to ONNX. For
   each (window, local speaker) pair, it maps that speaker's audio to one
   fixed-length vector, so that recordings of the same voice end up close
   together. TitaNet compares voices; it does not recognize words or
   identities.
3. **Clustering**: in sherpa-onnx 2.0.0, `FastClustering` normalizes the
   embeddings, computes cosine dissimilarity (1 minus cosine similarity), runs
   complete-linkage agglomerative clustering, and with `num_clusters=-1` cuts
   the tree at `clustering_threshold` (0.8 here). Each resulting cluster is
   one anonymous speaker for the whole recording; the number of speakers is
   inferred, never given.

This is sherpa-onnx's own reimplementation of the segmentation-embedding-
clustering recipe. It is not the complete upstream `pyannote.audio`
speaker-diarization pipeline, which uses a different embedding model and
clustering.

**The threshold trade-off.** A larger threshold merges more embeddings, so
there are fewer speakers; this can merge different people. A smaller
threshold can split one person across several labels. The official example
starts at 0.5. On held-out Act 3 of the benchmark play (measured here),
0.5 produced 97 clusters for 7 readers, 0.8 produced 27, and 1.0 produced 9,
but at 0.9 and above the two male voices of the 30-second test clip merged
into one. The default 0.8 was agreed as a compromise (specification,
section 9).

`min_duration_on` (0.3 s) and `min_duration_off` (0.5 s) are post-processing
settings from the official example; they were not tuned.

### 2.5 Joining words and speakers (`attribution.py`)

This step is plain, deterministic Python, fully unit-tested
(`tests/test_attribution.py`):

1. Overlapping turns of the same speaker are merged, so no time is counted
   twice.
2. Each word goes to the speaker whose turns overlap the word's interval the
   most.
3. Equal overlaps are broken by the speaker's earliest turn start, then by the
   lower cluster ID. Only these ties set `ambiguous=True`; a word inside
   overlapping speech with a clear winner is not marked ambiguous.
4. A word that no turn overlaps is `UNKNOWN`; the previous speaker is not
   carried over. A zero-length word goes to the turn containing its time.

`paragraphs()` then groups consecutive words with the same label, starting a
new paragraph on a speaker change, a gap of more than 2 s, or when a
paragraph would exceed 30 s. Speakers are numbered `SPEAKER_01`,
`SPEAKER_02`, ... in order of first appearance. `rendering.py` escapes
Markdown characters and formats `[hh:mm:ss]` from the paragraph start. None of
this changes the recognized words.

## 3. What the benchmark numbers mean

`scripts/benchmark.py` reports two accuracy numbers. They measure different
things and should not be combined.

### 3.1 Word error rate (WER)

WER compares recognized words with a reference text:

```text
WER = (substitutions + deletions + insertions) / words in the reference
```

The script lowercases both texts, maps dashes and slashes to spaces, removes
punctuation except apostrophes inside words, and then lets JiWER align the
word sequences. JiWER's default transforms only collapse spaces and split
words. Because the reference (Acts 1 and 2 of the play) is longer than any
test recording, the script first finds the span where reference and
recognized text align (`aligned_window()`, using matching blocks of at least
four words) and scores only that span. The report shows both the trimmed and
the full reference length.

WER depends on the reference. The references here come from the published
Project Gutenberg text, not from a transcription of these recordings. A spot
check of the 10-minute alignment found about 15 errors caused by the
reference rather than by recognition: archaic spellings such as "any one",
British spellings, and stage directions the readers skipped. WER says nothing
about speaker labels.

### 3.2 The word-based speaker score

This is a custom metric for this project. It is **not** the diarization error
rate (DER) and **not** a speaker-attributed WER.

`score()` takes the JiWER alignment and keeps only reference words that were
matched or substituted by a recognized word and that have a known reader
(stage directions are excluded). Each kept pair is (cluster of the recognized
word, reader of the reference word). Deleted and inserted words are not in
the denominator, so ASR errors affect this score only indirectly. A word
labeled `UNKNOWN` counts as wrong.

Two mappings from clusters to readers are reported:

- **One-to-one** (`best_one_to_one()`): each reader may be matched with at
  most one cluster, and the best such assignment is found exactly by dynamic
  programming over subsets of readers. If one reader is split across three
  clusters, only one of them can count as correct, so fragmentation is
  penalized. This is the number checked against the 90% gate.
- **Many-to-one**: every cluster is mapped to its most frequent reader. This
  measures cluster purity and hides fragmentation entirely: 40 pure clusters
  for 7 readers would score 100%.

On the 60-minute recording at threshold 0.8 (measured here), the score was
74.1% one-to-one and 93.7% many-to-one, with 40 clusters for 7 readers. In
words: most clusters contain one voice, but most voices are split across
several clusters. That is consistent with over-clustering, but the gap could
also include errors from local segmentation, from turns that miss short
replies, or from words whose estimated timestamps fall into a neighbouring
turn (hypothesis; not separated by any experiment).

### 3.3 How standard diarization metrics differ

The diarization error rate, as defined in pyannote.metrics, is
`(false alarm + missed detection + confusion) / total reference speech`,
measured in **time**, after an optimal one-to-one mapping of reference and
hypothesis labels (Hungarian algorithm). It covers speech detection as well
as speaker confusion, needs time-stamped reference turns, and is independent
of the recognized words. The custom score here instead counts **words**, needs
only a speaker-tagged reference text, and ignores time outside recognized
words. pyannote.metrics also defines purity and coverage, which correspond
roughly to the many-to-one view and to its fragmentation counterpart.

### 3.4 Scope of the benchmark

All accuracy numbers come from one LibriVox reading of a play, in which each
reader recorded their part separately. Every change of speaker is therefore
also a change of microphone and room, which may make speakers easier to tell
apart than in a meeting, while the 7 voices reading theatrical dialogue may
make it harder in other ways (hypothesis). The results are not evidence of
accuracy on real meetings.

## 4. Annotated reading list

Read the items in this order; the first three explain how the words are
recognized.

1. **Robust Speech Recognition via Large-Scale Weak Supervision.** Alec
   Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya
   Sutskever (OpenAI). ICML 2023, PMLR 202:28492-28518. Peer-reviewed
   conference paper.
   <https://proceedings.mlr.press/v202/radford23a.html>
   Read the sections on the approach (data, model, multitask format) and on
   limitations. It explains the log-Mel input, the 30-second windows, the
   encoder-decoder and why the output is predicted text. Its limitations
   section discusses hallucinated text and repetition loops; fluent output is
   not proof of fidelity. Results for the multilingual models do not carry
   over to `small.en` on this machine.
   Code: `asr.py`.

2. **faster-whisper 1.2.1.** SYSTRAN. Open-source implementation and
   documentation (not peer-reviewed).
   <https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1>, in particular
   [`transcribe.py`](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)
   (`WhisperModel.transcribe`, `find_alignment`, `restore_speech_timestamps`)
   and
   [`vad.py`](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/vad.py).
   It reimplements Whisper inference on CTranslate2; it is not a different
   model. Its README benchmarks are for other hardware and are no guarantee
   for this machine.
   Code: `asr.py`, `audio.py`.

3. **CTranslate2 quantization.** OpenNMT. Runtime documentation.
   <https://opennmt.net/CTranslate2/quantization.html>
   Explains which weights become INT8, which layers stay in floating point,
   and how a requested compute type falls back when the CPU does not support
   it. The alignment used for word timestamps is in
   [`src/models/whisper.cc` at v4.8.3](https://github.com/OpenNMT/CTranslate2/blob/v4.8.3/src/models/whisper.cc)
   (median filter and DTW).
   Code: `compute_type` in `settings.py`.

4. **Silero VAD.** Silero Team. Open-source model and documentation.
   <https://github.com/snakers4/silero-vad>
   The speech/non-speech detector bundled with faster-whisper. Read it to see
   what VAD does and does not decide.
   Code: `vad_filter` in `settings.py`.

5. **Powerset multi-class cross entropy loss for neural speaker
   diarization.** Alexis Plaquet, Hervé Bredin. INTERSPEECH 2023,
   pp. 3222-3226, doi:10.21437/Interspeech.2023-205. Peer-reviewed conference
   paper.
   <https://www.isca-archive.org/interspeech_2023/plaquet23_interspeech.pdf>
   Explains why the segmentation model predicts a class for each combination
   of active local speakers (including two overlapping speakers) instead of
   independent yes/no outputs per speaker, and why this removes a detection
   threshold. Read it with the **model card** of
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   (MIT licence; the files are gated on Hugging Face, which is why this
   project downloads sherpa-onnx's ONNX export instead), which gives the
   10-second window, 7 classes and the "not on its own" caveat.
   Code: `segmentation_model` in `settings.py`.

6. **TitaNet: Neural Model for Speaker Representation with 1D Depth-Wise
   Separable Convolutions and Global Context.** Nithin Rao Koluguri, Taejin
   Park, Boris Ginsburg (NVIDIA). ICASSP 2022, peer-reviewed:
   <https://ieeexplore.ieee.org/document/9746806>. Open preprint:
   <https://arxiv.org/abs/2110.04410>.
   Explains how variable-length speech becomes a fixed-length speaker
   embedding; TitaNet-S has about 6M parameters. The paper's verification and
   diarization scores come from the paper's own setup and data; they do not
   transfer to this ONNX export, this clustering or these recordings.
   Code: `embedding_model` in `settings.py`.

7. **sherpa-onnx speaker diarization.** k2-fsa (Xiaomi). Documentation and
   example code (not peer-reviewed).
   [Model list](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html)
   (rolling documentation),
   [official Python example at v2.0.0](https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.0/python-api-examples/offline-speaker-diarization.py),
   and the implementation at v2.0.0:
   [`offline-speaker-diarization-pyannote-impl.h`](https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.0/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h)
   (windows, powerset decoding, embeddings) and
   [`fast-clustering.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.0/sherpa-onnx/csrc/fast-clustering.cc)
   (cosine distance, complete linkage, threshold cut). The documentation's
   rule of thumb, "a larger threshold results in fewer speakers", is the
   trade-off behind the threshold sweep in the specification.
   Code: `diarization.py`.

8. **JiWER 4.0.0.** Jitsi. Library documentation (not peer-reviewed).
   <https://jitsi.github.io/jiwer/usage/> and
   <https://github.com/jitsi/jiwer/tree/v4.0.0>.
   Read how `process_words` aligns words and what the alignment chunks
   (`equal`, `substitute`, `delete`, `insert`) contain; the speaker score is
   built on them.
   Code: `score()` in `scripts/benchmark.py`.

9. **pyannote.metrics.** Hervé Bredin. Library documentation (not
   peer-reviewed): <https://pyannote.github.io/pyannote-metrics/>, especially
   the [diarization reference](https://pyannote.github.io/pyannote-metrics/reference.html).
   Defines DER, purity, coverage and JER, the standard way to evaluate
   diarization in time. Use it to see what the project's word-based score
   leaves out.
