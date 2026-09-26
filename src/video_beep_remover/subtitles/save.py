"""Write subtitles to a file, converting them to the format its extension names."""

from pathlib import Path

import pysubs2

from video_beep_remover.errors import SubtitleError, UsageError

FORMATS = {".srt": "srt", ".ass": "ass", ".ssa": "ssa", ".vtt": "vtt", ".sub": "microdvd", ".txt": "tmp"}


def save_subtitles(text: str, path: Path, *, fps: float | None = None) -> None:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise UsageError(f"unknown subtitle format {path.suffix!r}: use one of {', '.join(sorted(FORMATS))}")
    try:
        subs = pysubs2.SSAFile.from_string(text, fps=fps)
        path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(path), format_=fmt, fps=fps)
    except (OSError, ValueError, pysubs2.exceptions.Pysubs2Error) as exc:
        raise SubtitleError(f"could not write {path}: {exc}") from exc
