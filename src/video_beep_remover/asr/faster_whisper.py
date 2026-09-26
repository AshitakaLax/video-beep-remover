"""faster-whisper backend (DESIGN.md §6.8). The library is imported lazily: it is slow to load."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from video_beep_remover.asr.base import ProgressCallback
from video_beep_remover.config.schema import TranscriptionConfig
from video_beep_remover.errors import DependencyError
from video_beep_remover.models import Word

log = logging.getLogger(__name__)


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


def resolve_model(config: TranscriptionConfig, *, strategy: str, language: str) -> ModelChoice:
    """Resolve "auto": large-v3-turbo on GPU and for short windows; small(.en) for whole tracks on CPU."""
    device = config.device
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    name = config.model
    if name == "auto":
        if device == "cuda" or strategy != "full":
            name = "large-v3-turbo"
        else:
            name = "small.en" if language.lower().startswith("en") else "small"
    compute_type = config.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return ModelChoice(name=name, device=device, compute_type=compute_type, batched=device == "cuda")


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

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        offset: float,
        language: str,
        prompt: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> Sequence[Word]:
        options: dict[str, Any] = {
            "language": language,
            "word_timestamps": True,
            "beam_size": self.beam_size,
            "condition_on_previous_text": False,
            "initial_prompt": prompt,
            "vad_filter": self.vad_filter,
        }
        if self._pipeline is not None:
            segments, _ = self._pipeline.transcribe(audio, batch_size=self.batch_size, **options)
        else:
            segments, _ = self._model.transcribe(audio, **options)
        words: list[Word] = []
        for segment in segments:  # a generator: decoding happens while iterating
            for word in segment.words or ():
                start = offset + max(0.0, float(word.start))
                words.append(
                    Word(word.word, start, max(start, offset + float(word.end)), float(word.probability))
                )
            if on_progress is not None:
                on_progress(float(segment.end))
        log.debug("transcribed %.1f s of audio into %d words", len(audio) / 16_000, len(words))
        return words
