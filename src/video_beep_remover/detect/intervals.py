"""Turn detections into the spans the renderer mutes (DESIGN.md §6.10)."""

from collections.abc import Iterable

from video_beep_remover.models import CensorInterval, Detection

MIN_INTERVAL_S = 0.05  # every detection mutes at least this much, even a zero-length word with no padding


def build_intervals(
    detections: Iterable[Detection],
    *,
    duration: float,
    pad_before: float,
    pad_after: float,
    min_duration: float,
    merge_gap: float,
) -> list[CensorInterval]:
    """Pad, extend to a minimum length, clamp to the file, then merge. The result is sorted and disjoint."""
    min_duration = max(min_duration, MIN_INTERVAL_S)
    spans: list[tuple[float, float]] = []
    for detection in detections:
        start = detection.start - pad_before
        end = max(detection.end, detection.start) + pad_after
        if end - start < min_duration:
            middle = (start + end) / 2
            start, end = middle - min_duration / 2, middle + min_duration / 2
        # Keep the minimum length when a span hits either end of the file.
        if start < 0:
            end, start = end - start, 0.0
        if end > duration:
            start, end = max(0.0, start - (end - duration)), duration
        if end > start:
            spans.append((start, end))

    spans.sort()
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [CensorInterval(start, end) for start, end in merged]
