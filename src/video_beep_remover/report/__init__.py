"""JSON report, EDL and review SRT outputs (DESIGN.md §6.12), and reading a report back for
`vbr render --report`."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from video_beep_remover.errors import UsageError
from video_beep_remover.models import CensorInterval, Detection

SCHEMA_VERSION = 1
_HOW = {"estimate": " (estimated from subtitles)", "cue": " (whole subtitle cue)"}


def edl_text(intervals: Sequence[CensorInterval]) -> str:
    """Kodi / MPlayer edit decision list: `start end 1`, where action 1 means mute."""
    return "".join(f"{i.start:.3f}\t{i.end:.3f}\t1\n" for i in intervals)


def srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def review_srt(
    intervals: Sequence[CensorInterval],
    detections: Sequence[Detection],
    *,
    shift: float = 0.0,
    notes: Sequence[str | None] = (),
    extra: Sequence[tuple[float, float, str]] = (),
) -> str:
    """Subtitles with one cue per muted span, naming what was heard there, to spot-check a cleaned file
    in a player. `shift` moves every cue (the output's timeline_shift). `notes` (one per detection) and
    `extra` cues (start, end, text) carry the context layer's verdicts (DESIGN.md §17.4)."""
    cues: list[tuple[float, float, str]] = []
    for interval in intervals:
        heard = [
            d.heard.strip()
            + _HOW.get(d.source, "")
            + (f" ({notes[i]})" if i < len(notes) and notes[i] else "")
            for i, d in enumerate(detections)
            if d.start < interval.end and interval.start < d.end
        ]
        cues.append(
            (
                interval.start,
                interval.end,
                "[muted] " + ", ".join(dict.fromkeys(heard)) if heard else "[muted]",
            )
        )
    cues += extra
    blocks = []
    for number, (start, end, text) in enumerate(sorted(cues, key=lambda c: (c[0], c[1])), 1):
        blocks.append(f"{number}\n{srt_time(start + shift)} --> {srt_time(end + shift)}\n{text}\n")
    return "\n".join(blocks)


def detection_dict(detection: Detection) -> dict[str, Any]:
    data = asdict(detection)
    data["start"] = round(detection.start, 3)
    data["end"] = round(detection.end, 3)
    data["confidence"] = round(detection.confidence, 3)
    return data


def interval_dict(interval: CensorInterval) -> dict[str, float]:
    return {"start": round(interval.start, 3), "end": round(interval.end, 3)}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")


def read_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text("utf-8"))
    except OSError as exc:
        raise UsageError(f"cannot read the report {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise UsageError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError(f"{path} is not a vbr report")
    return data


def report_intervals(data: dict[str, Any]) -> list[CensorInterval]:
    """The report's `intervals`, which may have been edited by hand: sorted, overlapping ones merged."""
    raw = data.get("intervals")
    if not isinstance(raw, list):
        raise UsageError('the report has no "intervals" list')
    spans: list[tuple[float, float]] = []
    for number, item in enumerate(raw):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (TypeError, KeyError, ValueError) as exc:
            raise UsageError(f'intervals[{number}]: needs a numeric "start" and "end", got {item!r}') from exc
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start or end <= 0:
            raise UsageError(f"intervals[{number}]: end must be after start, and after 0 s: {item!r}")
        spans.append((max(0.0, start), end))
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [CensorInterval(start, end) for start, end in merged]


def report_detections(data: dict[str, Any]) -> list[Detection]:
    """The report's `detections`, as far as they are readable; they only label the review SRT."""
    return [detection for detection, _ in report_detection_items(data)]


def report_detection_items(data: dict[str, Any]) -> list[tuple[Detection, dict[str, Any]]]:
    """The readable detections of a report, each with its entry (which may hold a context verdict)."""
    found = []
    for item in data.get("detections") or []:
        try:
            source = item.get("source")
            cue = item.get("cue")
            detection = Detection(
                start=float(item["start"]),
                end=float(item["end"]),
                heard=str(item.get("heard") or ""),
                term=str(item.get("term") or ""),
                category=str(item.get("category") or ""),
                confidence=float(item.get("confidence") or 0.0),
                source=source if source in ("asr", "estimate", "cue") else "asr",
                cue=cue if isinstance(cue, int) else None,
            )
        except (AttributeError, TypeError, KeyError, ValueError):
            continue
        found.append((detection, item))
    return found
