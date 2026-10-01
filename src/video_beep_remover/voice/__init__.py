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

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from video_beep_remover.config.schema import ReplaceConfig
from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.detect.matcher import detect_in_words
from video_beep_remover.detect.normalize import normalize_token, split_words
from video_beep_remover.errors import DependencyError
from video_beep_remover.media.audio import Audio, read_pcm_windows
from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import StreamInfo
from video_beep_remover.models import CensorInterval, Detection, Word
from video_beep_remover.ui import UI
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

__all__ = ["Candidate", "Replacement", "Replacer", "VoiceModels", "check_installed", "replace_words"]

log = logging.getLogger(__name__)

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

    def release(self, name: str) -> bool:
        """Drop the separation model ("separator") or the voice model ("editor"), so its memory can be
        freed; its next use loads it again. Returns whether it was loaded. The speaker encoder is small
        and stays."""
        loaded = False
        if name == "separator":
            loaded, self._separator = self._separator is not None, None
        elif name == "editor":
            loaded, self._editor = self._editor is not None, None
        return loaded


def _tokens(text: str) -> list[str]:
    return [t for t in (normalize_token(raw) for raw, _, _ in split_words(text)) if t]


def _said(substitute: str, words: Sequence[Word]) -> bool:
    """Whether every token of the substitute was heard among `words`."""
    heard = [t for w in words for t in _tokens(w.text)]
    return all(any(fuzz.ratio(token, h) >= _MATCH for h in heard) for token in _tokens(substitute))


def _cosine(a: FloatArray, b: FloatArray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def _muted(candidate: Candidate, reason: str, **found: Any) -> Replacement:
    span = candidate.span
    word = candidate.detection.heard.strip()
    return Replacement(candidate.index, span[0], span[1], word, candidate.substitute, False, reason, **found)


def _guarded(step: "Callable[[_Work, StreamInfo, Path], None]", item: "_Work", *args: Any) -> None:
    """Run one stage for one word; a model that fails on it mutes the word. A model that cannot be
    loaded at all (DependencyError) stops the run."""
    try:
        step(item, *args)
    except DependencyError:
        raise
    except Exception as exc:
        log.warning("voice replacement failed at %.2f s", item.candidate.span[0], exc_info=True)
        item.result = _muted(item.candidate, f"voice replacement failed: {type(exc).__name__}: {exc}")


@dataclass
class _Work:
    """One word on its way through the stages of Replacer.replace."""

    candidate: Candidate
    window: FloatArray  # channels × samples at the stream's rate
    rows: list[int] = field(default_factory=list)  # the channels that carry dialogue
    vocals: FloatArray | None = None  # the separated voice in those channels
    edited: FloatArray | None = None  # the voice with the word said again
    delta: FloatArray | None = None  # what to add to the window
    result: Replacement | None = None  # once decided

    @property
    def local(self) -> tuple[float, float]:
        """The word's span, in seconds into the window."""
        start = self.candidate.window[0]
        return self.candidate.span[0] - start, self.candidate.span[1] - start


class Replacer:
    def __init__(
        self,
        config: ReplaceConfig,
        lexicon: Lexicon,
        ff: FFmpeg,
        models: VoiceModels,
        transcribe: Callable[[Audio, float], Sequence[Word]],
        make_room: Callable[[str], None] = lambda model: None,
    ) -> None:
        """`transcribe` hears 16 kHz mono audio that starts at the given media time. `make_room` is
        called with "separator" or "editor" before that model runs, so that the caller can free the
        others (models.keep_loaded)."""
        self.config = config
        self.lexicon = lexicon
        self.ff = ff
        self.models = models
        self.transcribe = transcribe
        self.make_room = make_room

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
        self,
        candidates: Sequence[Candidate],
        windows: Sequence[FloatArray],
        stream: StreamInfo,
        workdir: Path,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[Replacement]:
        """Say each candidate's word again in its window (the track over `candidate.window`, channels ×
        samples at the stream's rate), or say why it stays muted.

        Each model runs over every word before the next one starts: the separation model, the voice
        model, then the checks with speech recognition and the speaker encoder. So only one of the
        large models needs to be in memory at a time (models.keep_loaded, DESIGN.md §16). A model that
        fails on a word, e.g. out of GPU memory, mutes that word; one that cannot be loaded at all stops
        the run. `on_progress` receives the steps done so far, three per candidate."""
        rate = stream.sample_rate or 48_000
        work = [_Work(c, w) for c, w in zip(candidates, windows, strict=True)]
        for item in work:
            if item.window.shape[1] < round(item.local[1] * rate):
                item.result = _muted(item.candidate, "the audio around the word could not be read")
        done = 0
        stages = (("separator", self._separate), ("editor", self._edit), ("check", self._check))
        for model, step in stages:
            if model != "check" and any(item.result is None for item in work):
                self.make_room(model)
            for item in work:
                if item.result is None:
                    _guarded(step, item, stream, workdir)
                done += 1
                if on_progress is not None:
                    on_progress(done)
        results = [item.result for item in work]
        assert all(result is not None for result in results)  # the check decides every word left
        return [result for result in results if result is not None]

    def _separate(self, item: _Work, stream: StreamInfo, workdir: Path) -> None:
        item.rows = splice.dialogue_channels(stream.channel_layout, item.window.shape[0])
        item.vocals = self.models.separator().vocals(item.window[item.rows], stream.sample_rate or 48_000)

    def _edit(self, item: _Work, stream: StreamInfo, workdir: Path) -> None:
        assert item.vocals is not None
        rate = stream.sample_rate or 48_000
        voice = item.vocals.mean(axis=0)
        item.edited = self.models.editor().edit(voice, rate, item.candidate.spoken.text, item.local)
        item.delta = splice.change(item.window, item.vocals, item.edited, item.local, rate, item.rows)

    def _check(self, item: _Work, stream: StreamInfo, workdir: Path) -> None:
        assert item.vocals is not None and item.edited is not None and item.delta is not None
        candidate, rate = item.candidate, stream.sample_rate or 48_000
        span, substitute, start, local = candidate.span, candidate.substitute, candidate.window[0], item.local

        def muted(reason: str, **found: Any) -> None:
            item.result = _muted(candidate, reason, **found)

        mixed = self._to_16k(item.window + item.delta, rate, stream.channel_layout, workdir)
        words = self.transcribe(mixed, start)
        near = [w for w in words if span[0] <= (w.start + w.end) / 2 <= span[1]]  # the span is padded
        text = " ".join(w.text.strip() for w in near)
        leaked = [d for d in detect_in_words(self.lexicon, words) if d.start < span[1] and span[0] < d.end]
        if leaked:
            return muted(f"a listed word is still heard: {leaked[0].heard.strip()!r}", heard=text)
        if not _said(substitute, near):
            return muted(f"heard {text!r} instead of {substitute!r}", heard=text)
        similarity, baseline = self._likeness(item.vocals.mean(axis=0), item.edited, local, rate)
        if similarity < baseline - self.config.voice_margin:
            return muted(
                "the new word does not sound like the speaker",
                heard=text,
                similarity=similarity,
                baseline=baseline,
            )

        first, last = round(local[0] * rate), round(local[1] * rate)
        path = workdir / f"replace-{candidate.index}.f32"
        path.write_bytes(np.ascontiguousarray(item.delta[:, first:last].T, dtype=np.float32).tobytes())
        item.result = Replacement(
            candidate.index, span[0], span[1], candidate.detection.heard.strip(), substitute, True,
            "replaced", text, similarity, baseline, path, start + first / rate,
        )  # fmt: skip
        item.vocals = item.edited = item.delta = None

    def _to_16k(self, audio: FloatArray, rate: int, layout: str | None, workdir: Path) -> Audio:
        """`audio` (channels × samples) as the analysis hears a track: FFmpeg's downmix at 16 kHz mono,
        which keeps the front centre at full level. An even mix of a 5.1 track's channels would make the
        music 3–6 dB louder against the dialogue than the analysis heard it."""
        raw = workdir / "replace-check.f32"
        raw.write_bytes(np.ascontiguousarray(audio.T, dtype=np.float32).tobytes())
        channels = ["-ac", str(audio.shape[0]), *(["-ch_layout", layout] if layout else [])]
        try:
            data = self.ff.capture([
                "-f", "f32le", "-ar", str(rate), *channels, "-i", file_arg(raw),
                "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
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


def replace_words(
    replacer: Replacer,
    detections: Sequence[Detection],
    substitutes: Sequence[str | None],
    to_mute: Sequence[Detection],
    intervals: Sequence[CensorInterval],
    heard: Sequence[Word],
    *,
    source: Path,
    stream: StreamInfo,
    duration: float,
    workdir: Path,
    ui: UI,
) -> list[Replacement]:
    """Say a milder word in place of each detection that has a substitute (one per detection, or None)
    and is muted (`to_mute`, in `intervals`): a replacement for each, said again or still muted, and why.
    A word's span is replaced only if no other muted word shares it; any failure mutes it."""
    candidates = []
    for index, (detection, substitute) in enumerate(zip(detections, substitutes, strict=True)):
        if substitute is None or detection not in to_mute:
            continue
        middle = (detection.start + detection.end) / 2
        span = next((i for i in intervals if i.start <= middle <= i.end), None)
        if span is None:
            continue
        crowded = any(d != detection and d.start < span.end and span.start < d.end for d in to_mute)
        candidates.append((index, detection, substitute, span, crowded))
    planned: list[Candidate | Replacement] = []
    for index, detection, substitute, span, crowded in candidates:
        if crowded:
            planned.append(
                Replacement(
                    index, span.start, span.end, detection.heard.strip(), substitute, False,
                    "another muted word shares its span",
                )
            )  # fmt: skip
        else:
            planned.append(
                replacer.plan(
                    index=index,
                    detection=detection,
                    span=(span.start, span.end),
                    substitute=substitute,
                    heard=heard,
                    duration=duration,
                )
            )
    todo = [c for c in planned if isinstance(c, Candidate)]
    said_again: list[Replacement] = []
    if todo:
        windows = read_pcm_windows(replacer.ff, source, stream, [c.window for c in todo])
        with ui.progress("Replacing words", 3 * len(todo)) as update:
            said_again = replacer.replace(todo, windows, stream, workdir, on_progress=update)
    results = iter(said_again)
    done = [next(results) if isinstance(item, Candidate) else item for item in planned]
    if candidates:
        replaced = sum(r.replaced for r in done)
        reasons = sorted({r.reason for r in done if not r.replaced})
        ui.info(
            f"Voice replacement: {replaced} of {len(done)} words said again"
            + (f"; the rest stay muted ({'; '.join(reasons)})" if reasons else "")
        )
    return done
