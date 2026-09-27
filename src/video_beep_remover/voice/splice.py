"""Array helpers for voice replacement (DESIGN.md §16): the sentence a word was heard in, its text with
the substitute, the channels that carry dialogue, and the change to add to the track."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from video_beep_remover.models import Detection, Word

FloatArray = npt.NDArray[np.float32]

FADE_S = 0.02  # the change fades in and out over this much at the edges of the span
CONTEXT_S = 6.0  # at most this much of the sentence on either side of the word
SENTENCE_GAP_S = 1.0  # a pause this long ends a sentence
_SENTENCE_END = (".", "!", "?", "…")
_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
# Layouts whose third channel is the front centre, which carries most of a film's dialogue.
_CENTRE_LAYOUTS = {
    "3.0", "3.1", "4.0", "4.1", "5.0", "5.0(side)", "5.1", "5.1(side)", "6.0", "6.1", "6.1(back)",
    "7.0", "7.1", "7.1(wide)", "7.1(wide-side)",
}  # fmt: skip


@dataclass(frozen=True)
class Utterance:
    start: float  # media time of its first word
    end: float  # and of the end of its last
    original: str  # as heard
    text: str  # with the substitute in place of the word


def _middle(word: Word) -> float:
    return (word.start + word.end) / 2


def fit_case(heard: str, substitute: str) -> str:
    """The substitute in the case of the word it replaces, with that word's punctuation around it."""
    match = _EDGE.match(heard.strip())
    lead, core, trail = match.groups() if match else ("", heard.strip(), "")
    if len(core) > 1 and core.isupper():
        substitute = substitute.upper()
    elif core[:1].isupper():
        substitute = substitute[:1].upper() + substitute[1:]
    return f"{lead}{substitute}{trail}"


def utterance(heard: Sequence[Word], detection: Detection, substitute: str) -> Utterance | None:
    """The sentence the detection was heard in, with the substitute in its place; None if no heard word
    lies in the detection (a word only estimated from subtitles cannot be replaced)."""
    near = sorted(
        (w for w in heard if detection.start - CONTEXT_S <= _middle(w) <= detection.end + CONTEXT_S),
        key=lambda w: (w.start, w.end),
    )
    sentences: list[list[Word]] = [[]]
    for word in near:
        if sentences[-1] and word.start - sentences[-1][-1].end >= SENTENCE_GAP_S:
            sentences.append([])
        sentences[-1].append(word)
        if word.text.strip().endswith(_SENTENCE_END):
            sentences.append([])

    def inside(word: Word) -> bool:
        return detection.start - 0.05 <= _middle(word) <= detection.end + 0.05

    words = next((s for s in sentences if any(inside(w) for w in s)), None)
    if words is None:
        return None
    replaced = [w for w in words if inside(w)]
    first, last = (_EDGE.match(w.text.strip()) for w in (replaced[0], replaced[-1]))
    assert first is not None and last is not None  # _EDGE matches any string
    new = fit_case(first.group(1) + first.group(2) + last.group(3), substitute)
    parts = [new if w is replaced[0] else w.text.strip() for w in words if w is replaced[0] or not inside(w)]
    return Utterance(
        start=words[0].start,
        end=words[-1].end,
        original=" ".join(w.text.strip() for w in words),
        text=" ".join(part for part in parts if part),
    )


def dialogue_channels(layout: str | None, channels: int) -> list[int]:
    """The channels to edit: the front centre of a surround layout, the front pair otherwise."""
    if channels >= 3 and layout in _CENTRE_LAYOUTS:
        return [2]
    return [0, 1] if channels >= 2 else [0]


def fade_mask(length: int, rate: int, start: float, end: float, fade: float = FADE_S) -> FloatArray:
    """0 outside [start, end] (seconds into the window), 1 inside, with raised-cosine ramps of `fade`
    just inside each edge."""
    mask = np.zeros(length, dtype=np.float32)
    first, last = max(0, round(start * rate)), min(length, round(end * rate))
    if last <= first:
        return mask
    mask[first:last] = 1.0
    ramp = min(round(fade * rate), (last - first) // 2)
    if ramp > 0:
        curve = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp, endpoint=False))).astype(np.float32)
        mask[first : first + ramp] = curve
        mask[last - ramp : last] = curve[::-1]
    return mask


def _rms(samples: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0


def change(
    window: FloatArray,
    vocals: FloatArray,
    edited: FloatArray,
    span: tuple[float, float],
    rate: int,
    channels: Sequence[int],
) -> FloatArray:
    """What to add to `window` (channels × samples) so that, within `span`, its dialogue becomes
    `edited`: the edited voice minus the separated one, faded in and out at the span's edges and
    spread over the dialogue channels as the original voice was."""
    voice = vocals.mean(axis=0)
    mask = fade_mask(window.shape[1], rate, *span)
    difference = (edited[: voice.size] - voice) * mask[: voice.size]
    inside = slice(max(0, round(span[0] * rate)), round(span[1] * rate))
    level = _rms(voice[inside]) or _rms(voice)
    delta = np.zeros_like(window)
    for row, channel in enumerate(channels):
        gain = _rms(vocals[row][inside]) / level if level > 0 else 1.0
        delta[channel, : difference.size] = gain * difference
    return delta
