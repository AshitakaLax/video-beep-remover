"""JSON report and EDL outputs (DESIGN.md §6.12)."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from video_beep_remover.models import CensorInterval, Detection

SCHEMA_VERSION = 1


def edl_text(intervals: Sequence[CensorInterval]) -> str:
    """Kodi / MPlayer edit decision list: `start end 1`, where action 1 means mute."""
    return "".join(f"{i.start:.3f}\t{i.end:.3f}\t1\n" for i in intervals)


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
