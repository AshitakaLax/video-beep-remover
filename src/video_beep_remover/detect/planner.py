"""Flag subtitle cues and plan the audio windows to transcribe (DESIGN.md §6.7)."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.detect.matcher import Token, find_hints, find_matches, runs
from video_beep_remover.detect.normalize import split_words
from video_beep_remover.models import Cue, SyncModel, Window

STRONG_REASONS = frozenset({"lexicon", "masked"})
SPLIT_OVERLAP_S = 2.0  # pieces of a split window overlap by this much
UNCOVERED_MARGIN_S = 0.5  # speech this close to a cue counts as covered by it
MIN_UNCOVERED_S = 0.5  # shorter stretches of uncovered speech are ignored


@dataclass(frozen=True)
class FlaggedWord:
    start: int  # character span in the cue text
    end: int
    term: str
    category: str


@dataclass(frozen=True)
class FlaggedCue:
    cue: Cue
    reasons: frozenset[str]  # "lexicon" and "masked" are strong, "hint" is weak
    words: tuple[FlaggedWord, ...]  # the listed or masked words; empty for a hint-only flag

    @property
    def strong(self) -> bool:
        return bool(self.reasons & STRONG_REASONS)


def flag_cues(lexicon: Lexicon, cues: Iterable[Cue]) -> list[FlaggedCue]:
    """Cues that contain a listed term, a masked word or a hint word (DESIGN.md §6.7)."""
    flagged = []
    for cue in cues:
        spans = split_words(cue.text)
        tokens = [Token.from_raw(raw) for raw, _, _ in spans]
        reasons: set[str] = set()
        words: list[FlaggedWord] = []
        for match in find_matches(lexicon, tokens):
            reasons.add("masked" if match.masked else "lexicon")
            for run in runs(match.targets):
                words.append(FlaggedWord(spans[run[0]][1], spans[run[-1]][2], match.term, match.category))
        if find_hints(lexicon, tokens):
            reasons.add("hint")
        if reasons:
            flagged.append(FlaggedCue(cue, frozenset(reasons), tuple(words)))
    return flagged


def cue_span(cue: Cue, sync: SyncModel, pad: float = 0.0) -> tuple[float, float]:
    """The cue on the media timeline, widened by `pad` on both sides."""
    return sync.to_media(cue.start) - pad, sync.to_media(cue.end) + pad


def search_padding(window_padding: float, sync: SyncModel) -> float:
    """How far around a cue to listen: window_padding_s + 3 × the sync error."""
    return window_padding + 3 * sync.error


def flagged_windows(flags: Iterable[FlaggedCue], sync: SyncModel, pad: float) -> list[Window]:
    windows = []
    for flag in flags:
        start, end = cue_span(flag.cue, sync, pad)
        windows.append(Window(start, end, flag.reasons, (flag.cue.index,)))
    return windows


def _split(start: float, end: float, max_window: float, overlap: float) -> list[tuple[float, float]]:
    length = end - start
    if length <= max_window:
        return [(start, end)]
    overlap = min(overlap, max_window / 4)
    count = math.ceil((length - overlap) / (max_window - overlap))
    step = (length - overlap) / count
    pieces = [(start + i * step, start + i * step + step + overlap) for i in range(count)]
    pieces[-1] = (pieces[-1][0], end)
    return pieces


def plan_windows(
    spans: Iterable[Window],
    *,
    duration: float,
    min_window: float,
    max_window: float,
    merge_gap: float,
    overlap: float = SPLIT_OVERLAP_S,
) -> list[Window]:
    """Turn raw spans into the windows to transcribe (DESIGN.md §6.7 steps 2-3).

    Each span is extended around its middle to `min_window`, because Whisper does poorly on very
    short clips, and clamped to the file; spans outside the file are dropped. Spans less than
    `merge_gap` apart merge. Windows longer
    than `max_window` are split into pieces that overlap by `overlap`: the batched pipeline only
    transcribes the first 30 s of a clip. Each piece keeps the reasons and cues of the spans it
    touches."""
    grown: list[Window] = []
    for span in spans:
        start, end = span.start, span.end
        if end <= 0 or start >= duration:
            continue  # nothing to hear there, e.g. a cue after the end of a truncated file
        if end - start < min_window:
            middle = (start + end) / 2
            start, end = middle - min_window / 2, middle + min_window / 2
        if start < 0:
            start, end = 0.0, end - start
        if end > duration:
            start, end = max(0.0, start - (end - duration)), duration
        if end > start:
            grown.append(Window(start, end, span.reasons, span.cues))
    grown.sort(key=lambda w: (w.start, w.end))

    groups: list[tuple[list[Window], float]] = []  # (spans, end of the merged window)
    for window in grown:
        if groups and window.start - groups[-1][1] < merge_gap:
            groups[-1][0].append(window)
            groups[-1] = (groups[-1][0], max(groups[-1][1], window.end))
        else:
            groups.append(([window], window.end))

    planned: list[Window] = []
    for group, group_end in groups:
        for start, end in _split(group[0].start, group_end, max_window, overlap):
            touching = [w for w in group if w.start < end and start < w.end]
            reasons = frozenset().union(*(w.reasons for w in touching))
            cues = tuple(sorted({cue for w in touching for cue in w.cues}))
            planned.append(Window(start, end, reasons, cues))
    return planned


def uncovered_speech(
    speech: Sequence[tuple[float, float]],
    cues: Sequence[Cue],
    sync: SyncModel,
    *,
    margin: float = UNCOVERED_MARGIN_S,
    min_length: float = MIN_UNCOVERED_S,
) -> list[tuple[float, float]]:
    """Speech that no cue covers: songs, background voices and lines the subtitles skip
    (DESIGN.md §6.7 step 4). Every cue counts, not just flagged ones."""
    covered: list[list[float]] = []
    for start, end in sorted(cue_span(cue, sync, margin) for cue in cues):
        if covered and start <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], end)
        else:
            covered.append([start, end])

    pieces: list[tuple[float, float]] = []
    first = 0  # covered stretches that end before the current speech region are never needed again
    for start, end in sorted(speech):
        while first < len(covered) and covered[first][1] <= start:
            first += 1
        position = start
        for low, high in covered[first:]:
            if low >= end:
                break
            if low > position:
                pieces.append((position, low))
            position = max(position, high)
        if position < end:
            pieces.append((position, end))
    return [(start, end) for start, end in pieces if end - start >= min_length]


def audio_seconds(windows: Iterable[Window]) -> float:
    return sum(window.duration for window in windows)
