"""Voice replacement (DESIGN.md §16, M8): say a milder word in place of a listed one, in the same voice.

For a word the context layer lets through (§17.6), the sentence it was heard in is read from the
track, and:

1. its dialogue is split from music and effects: the front-centre channel of a surround track or the
   front pair, then a separation model;
2. a voice model speaks the word's span again, with the substitute in the sentence's text;
3. the change, new dialogue minus old within the span, is what the renderer adds to the track;
4. the result is checked: speech recognition must hear the substitute in the span and no listed word,
   and the new word must sound like the speaker about as much as the old one did (`voice_margin`).

A word that fails any step is muted, as it would have been without replacement. The span is muted in
every other audio stream, in the EDL, and by `vbr render`, which cannot replace words."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from video_beep_remover.config.schema import ReplaceConfig
from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.detect.matcher import detect_in_words
from video_beep_remover.detect.normalize import normalize_token, split_words
from video_beep_remover.media.audio import Audio
from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import StreamInfo
from video_beep_remover.models import Detection, Word
from video_beep_remover.voice import splice
from video_beep_remover.voice.models import (
    DemucsSeparator,
    EcapaEncoder,
    Editor,
    F5Editor,
    Separator,
    SpeakerEncoder,
    check_installed,
)
from video_beep_remover.voice.splice import FloatArray

__all__ = ["Candidate", "Replacement", "Replacer", "VoiceModels", "check_installed"]

MARGIN_S = 0.3  # of the track read before and after the sentence
MIN_REFERENCE_S = 1.0  # of the speaker's voice outside the span, to compare the new word with
_MATCH = 80  # rapidfuzz ratio from which a heard token counts as the substitute's


@dataclass(frozen=True)
class Replacement:
    detection: int  # index into the file's detections
    start: float  # the span replaced: the word's muted span
    end: float
    word: str
    substitute: str
    replaced: bool
    reason: str  # "replaced", or why the word stays muted
    heard: str = ""  # what speech recognition heard in the span afterwards
    similarity: float | None = None  # of the new word to the speaker's voice
    baseline: float | None = None  # of the old word to it
    delta: Path | None = None  # what the renderer adds: float32, interleaved, the stream's rate and channels
    delta_start: float = 0.0  # media time of the delta's first sample

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "word": self.word,
            "substitute": self.substitute,
            "replaced": self.replaced,
            "reason": self.reason,
            "heard": self.heard,
            "similarity": None if self.similarity is None else round(self.similarity, 3),
            "baseline": None if self.baseline is None else round(self.baseline, 3),
        }


@dataclass(frozen=True)
class Candidate:
    """A word to say again: the sentence with its substitute, and the window of the track to read."""

    index: int  # into the file's detections
    detection: Detection
    span: tuple[float, float]  # the word's muted span
    substitute: str
    spoken: splice.Utterance
    window: tuple[float, float]  # media time


class VoiceModels:
    """The separation model, the voice model and the speaker encoder, each loaded on first use and
    shared by every file of a batch. The factories replace them in tests."""

    def __init__(
        self,
        config: ReplaceConfig,
        *,
        device: str,
        offline: bool,
        cache_dir: Path,
        separator: Callable[[], Separator] | None = None,
        editor: Callable[[], Editor] | None = None,
        encoder: Callable[[], SpeakerEncoder] | None = None,
    ) -> None:
        self._make_separator = separator or (
            lambda: DemucsSeparator(config.separation, device=device, offline=offline)
        )
        self._make_editor = editor or (
            lambda: F5Editor(config.model, device=device, offline=offline, steps=config.steps)
        )
        self._make_encoder = encoder or (lambda: EcapaEncoder(cache_dir, device=device, offline=offline))
        self._separator: Separator | None = None
        self._editor: Editor | None = None
        self._encoder: SpeakerEncoder | None = None

    def separator(self) -> Separator:
        if self._separator is None:
            self._separator = self._make_separator()
        return self._separator

    def editor(self) -> Editor:
        if self._editor is None:
            self._editor = self._make_editor()
        return self._editor

    def encoder(self) -> SpeakerEncoder:
        if self._encoder is None:
            self._encoder = self._make_encoder()
        return self._encoder


def _tokens(text: str) -> list[str]:
    return [t for t in (normalize_token(raw) for raw, _, _ in split_words(text)) if t]


def _said(substitute: str, words: Sequence[Word]) -> bool:
    """Whether every token of the substitute was heard among `words`."""
    heard = [t for w in words for t in _tokens(w.text)]
    return all(any(fuzz.ratio(token, h) >= _MATCH for h in heard) for token in _tokens(substitute))


def _cosine(a: FloatArray, b: FloatArray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


class Replacer:
    def __init__(
        self,
        config: ReplaceConfig,
        lexicon: Lexicon,
        ff: FFmpeg,
        models: VoiceModels,
        transcribe: Callable[[Audio, float], Sequence[Word]],
    ) -> None:
        """`transcribe` hears 16 kHz mono audio that starts at the given media time."""
        self.config = config
        self.lexicon = lexicon
        self.ff = ff
        self.models = models
        self.transcribe = transcribe

    def plan(
        self,
        *,
        index: int,
        detection: Detection,
        span: tuple[float, float],
        substitute: str,
        heard: Sequence[Word],
        duration: float,
    ) -> "Candidate | Replacement":
        """The sentence to say again and the window of the track to read for it; or, when the word
        cannot be replaced, why. `span` is the word's muted span; `heard` the words heard around it."""
        spoken = splice.utterance(heard, detection, substitute)
        if spoken is None:
            return Replacement(
                index, span[0], span[1], detection.heard.strip(), substitute, False,
                "the word was not heard, only estimated",
            )  # fmt: skip
        window = (
            max(0.0, min(spoken.start, span[0]) - MARGIN_S),
            min(duration, max(spoken.end, span[1]) + MARGIN_S),
        )
        return Candidate(index, detection, span, substitute, spoken, window)

    def replace(
        self, candidate: "Candidate", window: FloatArray, stream: StreamInfo, workdir: Path
    ) -> Replacement:
        """Say the candidate's word again in `window` (the track over `candidate.window`, channels ×
        samples at the stream's rate), or say why it stays muted."""
        index, span, substitute = candidate.index, candidate.span, candidate.substitute
        word = candidate.detection.heard.strip()

        def muted(reason: str, **found: Any) -> Replacement:
            return Replacement(index, span[0], span[1], word, substitute, False, reason, **found)

        rate = stream.sample_rate or 48_000
        start = candidate.window[0]
        local = (span[0] - start, span[1] - start)
        if window.shape[1] < round(local[1] * rate):
            return muted("the audio around the word could not be read")

        rows = splice.dialogue_channels(stream.channel_layout, window.shape[0])
        vocals = self.models.separator().vocals(window[rows], rate)
        voice = vocals.mean(axis=0)
        edited = self.models.editor().edit(voice, rate, candidate.spoken.text, local)
        delta = splice.change(window, vocals, edited, local, rate, rows)

        words = self.transcribe(self._to_16k((window + delta).mean(axis=0), rate, workdir), start)
        near = [w for w in words if span[0] <= (w.start + w.end) / 2 <= span[1]]  # the span is padded
        text = " ".join(w.text.strip() for w in near)
        leaked = [d for d in detect_in_words(self.lexicon, words) if d.start < span[1] and span[0] < d.end]
        if leaked:
            return muted(f"a listed word is still heard: {leaked[0].heard.strip()!r}", heard=text)
        if not _said(substitute, near):
            return muted(f"heard {text!r} instead of {substitute!r}", heard=text)
        similarity, baseline = self._likeness(voice, edited, local, rate)
        if similarity < baseline - self.config.voice_margin:
            return muted(
                "the new word does not sound like the speaker",
                heard=text,
                similarity=similarity,
                baseline=baseline,
            )

        first, last = round(local[0] * rate), round(local[1] * rate)
        path = workdir / f"replace-{index}.f32"
        path.write_bytes(np.ascontiguousarray(delta[:, first:last].T, dtype=np.float32).tobytes())
        return Replacement(
            index, span[0], span[1], word, substitute, True, "replaced", text,
            similarity, baseline, path, start + first / rate,
        )  # fmt: skip

    def _to_16k(self, mono: FloatArray, rate: int, workdir: Path) -> Audio:
        raw = workdir / "replace-check.f32"
        raw.write_bytes(np.ascontiguousarray(mono, dtype=np.float32).tobytes())
        try:
            data = self.ff.capture([
                "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", file_arg(raw),
                "-ar", "16000", "-f", "f32le", "pipe:1",
            ])  # fmt: skip
        finally:
            raw.unlink(missing_ok=True)
        return np.frombuffer(data, dtype=np.float32).copy()

    def _likeness(
        self, voice: FloatArray, edited: FloatArray, span: tuple[float, float], rate: int
    ) -> tuple[float, float]:
        """How much the new word and the old one sound like the rest of the speaker's sentence."""
        first, last = round(span[0] * rate), round(span[1] * rate)
        rest = np.concatenate([voice[:first], voice[last:]])
        reference = rest if rest.size >= MIN_REFERENCE_S * rate else voice
        encoder = self.models.encoder()
        speaker = encoder.embed(reference, rate)
        return (
            _cosine(encoder.embed(edited[first:last], rate), speaker),
            _cosine(encoder.embed(voice[first:last], rate), speaker),
        )
