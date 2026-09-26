"""WhisperX backend (DESIGN.md §6.8): faster-whisper transcribes, then wav2vec2 forced alignment
re-times every word. On the evaluation set (Appendix C), Whisper's word ends came up to 0.23 s early
and the aligned ones within 0.05 s; but aligned starts came up to 0.2 s late, where Whisper's were
early. A word censored late is heard, so each word keeps the earlier start and the later end.

WhisperX, with PyTorch, is an optional extra: pip install "video-beep-remover[align]". It is imported
only when this backend is chosen: importing PyTorch takes seconds."""

import logging
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from video_beep_remover.asr.base import Clip, ProgressCallback
from video_beep_remover.asr.faster_whisper import FasterWhisperTranscriber, Segment
from video_beep_remover.errors import ConfigError, DependencyError
from video_beep_remover.media.audio import SAMPLE_RATE
from video_beep_remover.models import Word

log = logging.getLogger(__name__)

INSTALL = 'pip install "video-beep-remover[align]"'
UNSPACED = frozenset({"ja", "zh"})  # aligned character by character (WhisperX LANGUAGES_WITHOUT_SPACES)
MARGIN_S = 0.2  # audio kept around each segment: Whisper's segment edges can cut its first or last word
MAX_SHIFT_S = 0.5  # an aligned edge further out than this from Whisper's is a misalignment: ignored
_SPACE = re.compile(r"\s+")


def _import() -> Any:
    # WhisperX logs to stdout unless its logger already has a handler; with one, its messages go
    # through this tool's logging instead.
    logger = logging.getLogger("whisperx")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    try:
        import whisperx
        from whisperx import alignment
    except ImportError as exc:
        raise DependencyError(f'transcription.backend = "whisperx" needs WhisperX: {INSTALL}') from exc
    return whisperx, alignment


def _sentence_data_present() -> bool:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
        return True
    except LookupError:
        return False


def _sentence_data(offline: bool) -> None:
    """WhisperX splits text into sentences with NLTK's Punkt tables, and fetches them quietly on
    first use; a failed fetch would then fail every segment. Fetch them up front instead."""
    import nltk

    if _sentence_data_present():
        return
    if not offline and nltk.download("punkt_tab", quiet=True) and _sentence_data_present():
        return
    hint = "; offline mode: download it once without --offline" if offline else ""
    raise DependencyError(
        f"WhisperX needs NLTK's punkt_tab data: python -m nltk.downloader punkt_tab{hint} "
        "(behind a proxy, NLTK downloads only with NLTK_ALLOW_PROXIED_URLOPEN=1)"
    )


def _model_name(alignment: Any, language: str, model: str) -> str:
    """The alignment model to use: `model`, or with "auto" WhisperX's default for the language."""
    if model != "auto":
        return model
    defaults = {
        **getattr(alignment, "DEFAULT_ALIGN_MODELS_HF", {}),
        **getattr(alignment, "DEFAULT_ALIGN_MODELS_TORCH", {}),
    }
    name = defaults.get(language)
    if not isinstance(name, str):
        raise ConfigError(
            f"WhisperX has no alignment model for language {language!r}: set "
            "transcription.align_model to a wav2vec2 model on Hugging Face fine-tuned for it"
        )
    return name


def _torchaudio_file(name: str) -> Path | None:
    """Where torchaudio keeps the weights of one of its pipelines, e.g. WAV2VEC2_ASR_BASE_960H, or
    None when `name` is not a torchaudio pipeline (a Hugging Face model, then)."""
    import torch
    import torchaudio

    bundle = getattr(torchaudio.pipelines, name, None) if name in torchaudio.pipelines.__all__ else None
    path = getattr(bundle, "_path", None)
    if not isinstance(path, str):
        return None
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(path).name


def _downloaded(name: str) -> bool:
    weights = _torchaudio_file(name)
    if weights is not None:
        return weights.is_file()
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(name, "config.json"), str)


def status(language: str, model: str) -> tuple[bool, str]:
    """For vbr doctor: whether the alignment model and NLTK's data are downloaded, and a description."""
    _, alignment = _import()
    name = _model_name(alignment, language, model)
    missing = [
        what
        for what, ok in ((name, _downloaded(name)), ("NLTK punkt_tab", _sentence_data_present()))
        if not ok
    ]
    if missing:
        return False, f"{name}; not downloaded yet: {', '.join(missing)} (the first run downloads them)"
    return True, f"{name} is downloaded"


class Aligner:
    """A wav2vec2 alignment model for one language."""

    def __init__(self, language: str, model: str, device: str, *, offline: bool) -> None:
        self.whisperx, alignment = _import()
        self.language = language
        self.device = device
        self.name = _model_name(alignment, language, model)
        _sentence_data(offline)
        if offline and not _downloaded(self.name):
            raise DependencyError(
                f"the alignment model {self.name} is not downloaded "
                "(offline mode: download it once without --offline)"
            )
        try:
            self.model, self.metadata = self.whisperx.load_align_model(
                language, device, model_name=self.name, model_cache_only=offline
            )
        except Exception as exc:
            hint = " (offline mode: download it once without --offline)" if offline else ""
            raise DependencyError(f"could not load the alignment model {self.name!r}{hint}: {exc}") from exc

    def retime(self, clip: Clip, segment: Segment) -> list[Word]:
        """The segment's words, widened to their aligned times. A word the aligner could not place
        keeps Whisper's times."""
        words = list(segment.words)
        spoken = [i for i, w in enumerate(words) if w.text.strip()]
        if not spoken:
            return words
        spaced = self.language not in UNSPACED
        # WhisperX aligns a segment's text split at spaces, or character by character.
        texts = [words[i].text.strip() if spaced else _SPACE.sub("", words[i].text) for i in spoken]
        units = [1 if spaced else len(text) for text in texts]
        start = min(segment.start, words[spoken[0]].start) - MARGIN_S
        end = max(segment.end, words[spoken[-1]].end) + MARGIN_S
        first = max(0, round((start - clip.start) * SAMPLE_RATE))
        last = min(len(clip.audio), round((end - clip.start) * SAMPLE_RATE))
        audio = np.array(clip.audio[first:last], dtype=np.float32)  # a copy: PyTorch wants writable memory
        offset = clip.start + first / SAMPLE_RATE
        text = (" " if spaced else "").join(texts)
        try:
            result = self.whisperx.align(
                [{"start": 0.0, "end": len(audio) / SAMPLE_RATE, "text": text}],
                self.model,
                self.metadata,
                audio,
                self.device,
            )
        except Exception as exc:  # alignment only refines the times: keep Whisper's rather than fail
            log.warning("could not align %r at %.1f s: %s", text, segment.start, exc)
            return words
        found = [w for part in result.get("segments", ()) for w in part.get("words", ())]
        if len(found) != sum(units):
            log.debug(
                "aligning %r gave %d words for %d; keeping Whisper's times", text, len(found), len(units)
            )
            return words
        position = 0
        for index, count in zip(spoken, units, strict=True):
            group = found[position : position + count]
            position += count
            starts = [t for t in (_seconds(w.get("start")) for w in group) if t is not None]
            ends = [t for t in (_seconds(w.get("end")) for w in group) if t is not None]
            if starts and ends and max(ends) >= min(starts):
                words[index] = _widened(words[index], offset + min(starts), offset + max(ends))
        return words


def _widened(word: Word, start: float, end: float) -> Word:
    """The word from the earlier start to the later end of Whisper's times and the aligned ones."""
    if word.start - start > MAX_SHIFT_S:
        start = word.start
    if end - word.end > MAX_SHIFT_S:
        end = word.end
    return Word(word.text, min(word.start, start), max(word.end, end), word.probability)


def _seconds(value: Any) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) else None


class WhisperXTranscriber:
    """faster-whisper words, re-timed by forced alignment segment by segment as they are decoded."""

    def __init__(
        self, whisper: FasterWhisperTranscriber, *, language: str, align_model: str, offline: bool
    ) -> None:
        self.whisper = whisper
        self.name = f"{whisper.name} with alignment"
        self.align_model = align_model
        self.offline = offline
        self._aligners: dict[str, Aligner] = {}
        self._aligner(language)  # fail now, not after transcribing, if the aligner cannot load

    def _aligner(self, language: str) -> Aligner:
        if language not in self._aligners:
            self._aligners[language] = Aligner(
                language, self.align_model, self.whisper.choice.device, offline=self.offline
            )
        return self._aligners[language]

    def transcribe(
        self,
        clips: Sequence[Clip],
        *,
        language: str,
        prompt: str | None,
        vad: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[Word]]:
        aligner = self._aligner(language)
        results: list[list[Word]] = [[] for _ in clips]
        segments: Iterator[Segment] = self.whisper.segments(
            clips, language=language, prompt=prompt, vad=vad, on_progress=on_progress
        )
        for segment in segments:
            results[segment.clip] += aligner.retime(clips[segment.clip], segment)
        return results
