"""Lines of dialogue for the context layer (DESIGN.md §17.2): subtitle cues on the media timeline,
and sentences of heard words where no cue covers the speech."""

import bisect
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from video_beep_remover.detect.normalize import normalize_token, split_words
from video_beep_remover.models import Cue, Detection, Sound, SyncModel, Word

SENTENCE_GAP_S = 1.0  # a pause this long ends a sentence of heard words
COVER_MARGIN_S = 0.5  # a heard word this close to a cue belongs to the cue
_SENTENCE_END = (".", "!", "?", "…")


@dataclass(frozen=True)
class Line:
    start: float  # media time
    end: float
    text: str  # what is said; empty for a cue that only describes a sound
    sounds: tuple[str, ...] = ()  # sound descriptions during the line, e.g. "moaning"
    cue: int | None = None  # the subtitle cue, if the line is one


def cue_lines(cues: Sequence[Cue], sounds: Sequence[Sound], sync: SyncModel) -> list[Line]:
    """Cues on the media timeline, each with the sound descriptions it overlaps. A sound that
    overlaps no cue (a cue that only says "[moaning]") becomes a line of its own."""
    attached: list[list[str]] = [[] for _ in cues]
    alone: list[Sound] = []
    for sound in sounds:
        hits = [i for i, cue in enumerate(cues) if sound.start < cue.end and cue.start < sound.end]
        for i in hits:
            attached[i].append(sound.text)
        if not hits:
            alone.append(sound)
    lines = [
        Line(
            sync.to_media(cue.start), sync.to_media(cue.end), cue.text, tuple(dict.fromkeys(extra)), cue.index
        )
        for cue, extra in zip(cues, attached, strict=True)
    ]
    lines += [Line(sync.to_media(s.start), sync.to_media(s.end), "", (s.text,)) for s in alone]
    return sorted(lines, key=lambda line: (line.start, line.end))


def word_lines(words: Sequence[Word]) -> list[Line]:
    """Sentences of heard words: a word ending in . ! ? or … ends one, and so does a pause of
    SENTENCE_GAP_S."""
    lines: list[Line] = []
    current: list[Word] = []

    def close() -> None:
        if current:
            text = " ".join(w.text.strip() for w in current if w.text.strip())
            lines.append(Line(current[0].start, current[-1].end, text))
            current.clear()

    for word in sorted(words, key=lambda w: (w.start, w.end)):
        if current and word.start - current[-1].end >= SENTENCE_GAP_S:
            close()
        current.append(word)
        if word.text.strip().endswith(_SENTENCE_END):
            close()
    close()
    return lines


def build_lines(
    cues: Sequence[Cue], sounds: Sequence[Sound], sync: SyncModel, heard: Sequence[Word]
) -> list[Line]:
    """The cues, plus sentences of the heard words no cue covers (hybrid's unsubtitled speech, or
    everything when there are no subtitles)."""
    lines = cue_lines(cues, sounds, sync)
    spans: list[list[float]] = []  # what the cues cover, merged: sorted and disjoint
    for line in lines:  # sorted by start
        if line.cue is None:
            continue
        start, end = line.start - COVER_MARGIN_S, line.end + COVER_MARGIN_S
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    starts = [start for start, _ in spans]

    def covered(word: Word) -> bool:
        middle = (word.start + word.end) / 2
        i = bisect.bisect_right(starts, middle) - 1
        return i >= 0 and middle <= spans[i][1]

    uncovered = [w for w in heard if not covered(w)]
    return sorted(lines + word_lines(uncovered), key=lambda line: (line.start, line.end))


def line_for(detection: Detection, lines: Sequence[Line]) -> int | None:
    """The line a detection belongs to: its subtitle cue, or else the line around its middle."""
    if detection.cue is not None:
        for index, line in enumerate(lines):
            if line.cue == detection.cue:
                return index
    middle = (detection.start + detection.end) / 2
    near = [
        (abs((line.start + line.end) / 2 - middle), index)
        for index, line in enumerate(lines)
        if line.text and line.start - COVER_MARGIN_S <= middle <= line.end + COVER_MARGIN_S
    ]
    return min(near)[1] if near else None


def neighbours(
    lines: Sequence[Line], index: int, shown: Callable[[int], bool] = lambda i: True
) -> tuple[str, str]:
    """The spoken lines just before and after a line, a sentence often spanning two cues; "" where
    that line is not to be `shown`."""

    def text(i: int | None) -> str:
        return lines[i].text if i is not None and shown(i) else ""

    before = next((i for i in range(index - 1, -1, -1) if lines[i].text), None)
    after = next((i for i in range(index + 1, len(lines)) if lines[i].text), None)
    return text(before), text(after)


def _tokens(text: str) -> list[str]:
    return [token for token in (normalize_token(raw) for raw, _, _ in split_words(text)) if token]


def heard_share(line: Line, heard: Sequence[Word]) -> float:
    """The share of a line's words that were heard around it. Subtitles can hold text that is never
    said, such as a note written for a model, so only lines that were heard are trusted to decide
    that a use is harmless (DESIGN.md §17.9)."""
    said = Counter(_tokens(line.text))
    if not said:
        return 0.0
    near = Counter(
        token
        for word in heard
        if line.start - COVER_MARGIN_S <= (word.start + word.end) / 2 <= line.end + COVER_MARGIN_S
        for token in _tokens(word.text)
    )
    return sum((said & near).values()) / sum(said.values())
