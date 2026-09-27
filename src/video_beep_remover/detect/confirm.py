"""Turn window transcripts into detections and check them against flagged cues (DESIGN.md §6.8-6.9)."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Literal

from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.detect.matcher import detect_in_words
from video_beep_remover.detect.planner import FlaggedCue, cue_span
from video_beep_remover.models import Detection, SyncModel, Window, Word

EDGE_S = 0.3  # words this close to a window edge are unreliable
ESTIMATE_PAD_S = 0.3  # an estimated word is censored this much wider on both sides
LOW_CONFIDENCE = 0.2  # confidence given to estimated and whole-cue detections
Resolution = Literal["estimate", "cue", "skip"]


def assemble(
    windows: Sequence[Window],
    transcripts: Sequence[Sequence[Word]],
    duration: float,
    clean_edges: Sequence[tuple[bool, bool]] | None = None,
) -> list[list[Word]]:
    """The words heard in each group of overlapping windows (the pieces of one split window), in time
    order. Words within EDGE_S of a group's outer edge are dropped: a window edge can cut a word in
    half. That does not apply at the start or end of the file, or at an edge marked clean in
    `clean_edges` ((start, end) per window), i.e. one trimmed to silence. Where two pieces overlap,
    each keeps its words before or after the middle of the overlap, so no word is counted twice."""
    order = sorted(range(len(windows)), key=lambda i: (windows[i].start, windows[i].end))
    groups: list[list[int]] = []
    for index in order:
        if groups and windows[index].start < windows[groups[-1][-1]].end:
            groups[-1].append(index)
        else:
            groups.append([index])

    heard: list[list[Word]] = []
    for group in groups:
        first, last = windows[group[0]], windows[group[-1]]
        clean_start = clean_edges[group[0]][0] if clean_edges else False
        clean_end = clean_edges[group[-1]][1] if clean_edges else False
        low = first.start + EDGE_S if first.start > 1e-6 and not clean_start else -math.inf
        high = last.end - EDGE_S if last.end < duration - 1e-6 and not clean_end else math.inf
        words: list[Word] = []
        for position, index in enumerate(group):
            window = windows[index]
            previous = windows[group[position - 1]] if position else None
            following = windows[group[position + 1]] if position + 1 < len(group) else None
            after = (window.start + previous.end) / 2 if previous else -math.inf
            before = (following.start + window.end) / 2 if following else math.inf
            words += [
                word
                for word in transcripts[index]
                if after <= word.start < before and word.start >= low and word.end <= high
            ]
        heard.append(sorted(words, key=lambda w: w.start))
    return heard


def detect_in_windows(
    lexicon: Lexicon,
    windows: Sequence[Window],
    transcripts: Sequence[Sequence[Word]],
    duration: float,
    clean_edges: Sequence[tuple[bool, bool]] | None = None,
) -> tuple[list[Detection], list[Word]]:
    """Detections in window transcripts, and the words they were found among. Each group of windows
    is matched on its own, so a phrase never spans two separate windows."""
    detections: list[Detection] = []
    heard: list[Word] = []
    for words in assemble(windows, transcripts, duration, clean_edges):
        heard += words
        detections += detect_in_words(lexicon, words)
    return detections, heard


def dedupe(detections: Iterable[Detection]) -> list[Detection]:
    """Drop repeats of the same term at the same time (from overlapping or re-transcribed windows),
    keeping the most confident one."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: (d.start, d.end)):
        for position, other in enumerate(kept):
            if other.term == detection.term and detection.start <= other.end and other.start <= detection.end:
                if detection.confidence > other.confidence:
                    kept[position] = detection
                break
        else:
            kept.append(detection)
    return sorted(kept, key=lambda d: (d.start, d.end))


def overlaps(detection: Detection, span: tuple[float, float]) -> bool:
    return detection.start <= span[1] and span[0] <= detection.end


def split_confirmed(
    flags: Iterable[FlaggedCue], detections: Sequence[Detection], sync: SyncModel, pad: float
) -> tuple[list[FlaggedCue], list[FlaggedCue]]:
    """(confirmed, unconfirmed) strong flags. A flag is confirmed when a detection overlaps the cue
    widened by the search padding (DESIGN.md §6.9)."""
    confirmed: list[FlaggedCue] = []
    unconfirmed: list[FlaggedCue] = []
    for flag in flags:
        if not flag.strong:
            continue
        span = cue_span(flag.cue, sync, pad)
        (confirmed if any(overlaps(d, span) for d in detections) else unconfirmed).append(flag)
    return confirmed, unconfirmed


def resolve_unconfirmed(
    flag: FlaggedCue, how: Resolution, sync: SyncModel, duration: float
) -> list[Detection]:
    """What to censor for a strong flag the audio did not confirm (`on_unconfirmed`).

    - estimate: each flagged word's time, from its position in the cue text (speech is roughly
      uniform within a cue), widened by ESTIMATE_PAD_S
    - cue: the whole cue
    - skip: nothing"""
    if how == "skip" or not flag.words:
        return []
    start, end = cue_span(flag.cue, sync)
    start, end = max(0.0, start), min(duration, end)
    if end <= start:
        return []
    text = flag.cue.text
    if how == "cue":
        word = flag.words[0]
        return [Detection(start, end, text, word.term, word.category, LOW_CONFIDENCE, "cue", flag.cue.index)]
    per_char = (end - start) / max(1, len(text))
    return [
        Detection(
            max(0.0, start + word.start * per_char - ESTIMATE_PAD_S),
            min(duration, start + word.end * per_char + ESTIMATE_PAD_S),
            text[word.start : word.end],
            word.term,
            word.category,
            LOW_CONFIDENCE,
            "estimate",
            flag.cue.index,
        )
        for word in flag.words
    ]


def attribute(
    detections: Iterable[Detection], flags: Sequence[FlaggedCue], sync: SyncModel, pad: float
) -> list[Detection]:
    """Record which flagged cue each speech-recognition detection belongs to, if any."""
    spans = [(flag.cue.index, cue_span(flag.cue, sync, pad)) for flag in flags]
    attributed = []
    for detection in detections:
        if detection.cue is None:
            index = next((cue for cue, span in spans if overlaps(detection, span)), None)
            detection = replace(detection, cue=index)
        attributed.append(detection)
    return attributed
