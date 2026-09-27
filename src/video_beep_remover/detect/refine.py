"""Move censor edges into quiet audio (DESIGN.md §6.10, censor.refine_edges).

Padding puts each edge 120–200 ms from the word, usually in the pause between words. When it
lands in speech, the fade cuts a syllable in half; moving the edge outward to the quietest 10 ms
nearby puts the fade in the dip between syllables or words instead. Edges only ever move outward,
so a refined interval still covers everything the padded one did."""

from collections.abc import Sequence

import numpy as np

from video_beep_remover.media.audio import SAMPLE_RATE, Audio, AudioSource
from video_beep_remover.models import CensorInterval

FRAME_S = 0.01  # energy is compared over frames this long, the length of the default fade
SEARCH_S = 0.08  # how far an edge may move
TIE = 10 ** (1 / 20)  # frames within 1 dB of the quietest count as quiet: the nearest one wins
SILENT = 10 ** (-60 / 20)  # and so do frames below -60 dBFS, however much quieter the quietest is
READ_GAP_S = 5.0  # intervals closer than this are read from the file in one piece


def _rms(audio: Audio, start: int, end: int) -> float:
    frame = audio[max(0, start) : max(0, end)]
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else float("inf")


def _quietest(audio: Audio, edge: int, step: int) -> int:
    """Samples to move the edge by (a multiple of `step`, whose sign is the outward direction): to
    the frame, within SEARCH_S, where the fade would be quietest. The fade runs from a start edge
    into the interval, and into an end edge from inside it."""
    frame = abs(step)
    candidates = []
    for k in range(round(SEARCH_S / FRAME_S) + 1):
        at = edge + k * step
        candidates.append(_rms(audio, at, at + frame) if step < 0 else _rms(audio, at - frame, at))
    floor = min(candidates)
    if not np.isfinite(floor):
        return 0
    k = next(k for k, rms in enumerate(candidates) if rms <= max(floor * TIE, SILENT))
    return k * step


def _groups(intervals: Sequence[CensorInterval]) -> list[list[CensorInterval]]:
    groups: list[list[CensorInterval]] = []
    for interval in intervals:
        if groups and interval.start - groups[-1][-1].end < READ_GAP_S:
            groups[-1].append(interval)
        else:
            groups.append([interval])
    return groups


def refine_edges(
    intervals: Sequence[CensorInterval], audio: AudioSource, *, duration: float, merge_gap: float
) -> list[CensorInterval]:
    """Each edge moved outward, by at most SEARCH_S, to where the audio is quietest; then clamped to
    the file and merged again. `intervals` must be sorted and disjoint, and so is the result."""
    frame = round(FRAME_S * SAMPLE_RATE)
    moved: list[tuple[float, float]] = []
    for group in _groups(intervals):
        origin = max(0.0, group[0].start - SEARCH_S - FRAME_S)
        samples = audio.read(origin, min(duration, group[-1].end + SEARCH_S + FRAME_S))
        for interval in group:
            first = round((interval.start - origin) * SAMPLE_RATE)
            last = round((interval.end - origin) * SAMPLE_RATE)
            moved.append(
                (
                    max(0.0, interval.start + _quietest(samples, first, -frame) / SAMPLE_RATE),
                    min(duration, interval.end + _quietest(samples, last, frame) / SAMPLE_RATE),
                )
            )
    merged: list[list[float]] = []
    for start, end in moved:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [CensorInterval(start, end) for start, end in merged]
