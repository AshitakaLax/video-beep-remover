"""faster-whisper backend (DESIGN.md §6.8). The library is imported lazily: it is slow to load."""

import bisect
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from video_beep_remover.asr.base import Clip, ProgressCallback
from video_beep_remover.config.schema import TranscriptionConfig
from video_beep_remover.errors import DependencyError
from video_beep_remover.media.audio import SAMPLE_RATE
from video_beep_remover.models import Word

log = logging.getLogger(__name__)

CHUNK_S = 30.0  # Whisper's context; the batched pipeline transcribes only this much of each clip
_CLIP_TOLERANCE_S = 0.01  # segment start times are rounded to milliseconds


def cuda_available() -> bool:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # no CUDA runtime, or ctranslate2 missing
        return False


@dataclass(frozen=True)
class ModelChoice:
    name: str
    device: str
    compute_type: str
    batched: bool  # BatchedInferencePipeline; used on GPU

    def describe(self) -> str:
        return f"{self.name} ({self.device}, {self.compute_type})"


def _device_and_precision(config: TranscriptionConfig) -> tuple[str, str]:
    device = config.device
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    compute_type = config.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def resolve_model(config: TranscriptionConfig, *, strategy: str, language: str) -> ModelChoice:
    """Resolve "auto": large-v3-turbo on GPU and for short windows; small(.en) for whole tracks on CPU."""
    device, compute_type = _device_and_precision(config)
    name = config.model
    if name == "auto":
        if device == "cuda" or strategy != "full":
            name = "large-v3-turbo"
        else:
            name = "small.en" if language.lower().startswith("en") else "small"
    return ModelChoice(name=name, device=device, compute_type=compute_type, batched=device == "cuda")


def resolve_anchor_model(config: TranscriptionConfig, *, language: str) -> ModelChoice:
    """The small model for sync anchors. English-only models (".en") fall back to their multilingual
    version for other languages. Anchors are transcribed one at a time, so never batched."""
    device, compute_type = _device_and_precision(config)
    name = config.anchor_model
    if name.endswith(".en") and not language.lower().startswith("en"):
        name = name.removesuffix(".en")
    return ModelChoice(name=name, device=device, compute_type=compute_type, batched=False)


class FasterWhisperTranscriber:
    def __init__(
        self,
        choice: ModelChoice,
        *,
        beam_size: int,
        batch_size: int,
        vad_filter: bool,
        offline: bool,
        download_root: Path | None = None,
    ) -> None:
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise DependencyError("faster-whisper is not installed: pip install faster-whisper") from exc
        self.name = choice.describe()
        self.choice = choice
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.vad_filter = vad_filter
        try:
            self._model = WhisperModel(
                choice.name,
                device=choice.device,
                compute_type=choice.compute_type,
                download_root=str(download_root) if download_root else None,
                local_files_only=offline,
            )
        except Exception as exc:
            hint = " (offline mode: download it once without --offline)" if offline else ""
            raise DependencyError(f"could not load Whisper model {choice.name!r}{hint}: {exc}") from exc
        self._pipeline = BatchedInferencePipeline(model=self._model) if choice.batched else None

    def _options(self, language: str, prompt: str | None, vad: bool) -> dict[str, Any]:
        return {
            "language": language,
            "word_timestamps": True,
            "beam_size": self.beam_size,
            "condition_on_previous_text": False,
            "initial_prompt": prompt,
            "vad_filter": vad,
        }

    def transcribe(
        self,
        clips: Sequence[Clip],
        *,
        language: str,
        prompt: str | None,
        vad: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[Word]]:
        options = self._options(language, prompt, vad)
        if self._pipeline is not None and len(clips) > 1 and all(c.duration <= CHUNK_S for c in clips):
            return self._transcribe_packed(clips, options, on_progress)
        results: list[list[Word]] = []
        done = 0.0
        for clip in clips:
            results.append(self._transcribe_clip(clip, options, done, on_progress))
            done += clip.duration
            if on_progress is not None:
                on_progress(done)
        return results

    def _transcribe_clip(
        self, clip: Clip, options: dict[str, Any], done: float, on_progress: ProgressCallback | None
    ) -> list[Word]:
        audio = np.ascontiguousarray(clip.audio, dtype=np.float32)
        if self._pipeline is not None:
            if clip.duration > CHUNK_S and not options["vad_filter"]:
                # Without VAD the batched pipeline needs explicit chunks of at most 30 s.
                options = options | {
                    "clip_timestamps": [
                        {"start": t, "end": min(t + CHUNK_S, clip.duration)}
                        for t in np.arange(0.0, clip.duration, CHUNK_S).tolist()
                    ]
                }
            segments, _ = self._pipeline.transcribe(audio, batch_size=self.batch_size, **options)
        else:
            segments, _ = self._model.transcribe(audio, **options)
        words: list[Word] = []
        for segment in segments:  # a generator: decoding happens while iterating
            words += _words(segment.words or (), shift=clip.start)
            if on_progress is not None:
                on_progress(done + float(segment.end))
        log.debug("transcribed %.1f s at %.1f s into %d words", clip.duration, clip.start, len(words))
        return words

    def _transcribe_packed(
        self, clips: Sequence[Clip], options: dict[str, Any], on_progress: ProgressCallback | None
    ) -> list[list[Word]]:
        """All clips in one buffer, one batch item each (DESIGN.md §6.8): the GPU decodes them together."""
        assert self._pipeline is not None
        starts: list[float] = []
        position = 0
        for clip in clips:
            starts.append(position / SAMPLE_RATE)
            position += len(clip.audio)
        buffer = np.concatenate([np.asarray(clip.audio, dtype=np.float32) for clip in clips])
        stamps = [
            {"start": start, "end": start + clip.duration} for start, clip in zip(starts, clips, strict=True)
        ]
        segments, _ = self._pipeline.transcribe(
            buffer, batch_size=self.batch_size, **(options | {"clip_timestamps": stamps})
        )
        results: list[list[Word]] = [[] for _ in clips]
        for segment in segments:
            index = max(0, bisect.bisect_right(starts, float(segment.start) + _CLIP_TOLERANCE_S) - 1)
            results[index] += _words(segment.words or (), shift=clips[index].start - starts[index])
            if on_progress is not None:
                on_progress(float(segment.end))
        return results


def _words(found: Iterable[Any], *, shift: float) -> list[Word]:
    """faster-whisper words, moved onto the media timeline."""
    words = []
    for word in found:
        start = shift + max(0.0, float(word.start))
        words.append(Word(word.word, start, max(start, shift + float(word.end)), float(word.probability)))
    return words
