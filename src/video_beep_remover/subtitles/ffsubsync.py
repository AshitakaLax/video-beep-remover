"""Re-sync subtitles with ffsubsync when the sync check fails (DESIGN.md §6.6 step 7).

ffsubsync (the optional [sync] extra) aligns subtitles to the speech activity in the audio. It
corrects offsets of up to a minute and frame-rate mismatches, which anchor tracking cannot follow.
It runs as a separate program, so it stays an optional dependency."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from video_beep_remover.errors import SubtitleError
from video_beep_remover.subtitles.parse import read_subtitle_file
from video_beep_remover.subtitles.save import save_subtitles

TIMEOUT_S = 900


def ffsubsync_command() -> list[str] | None:
    """How to run ffsubsync, if it is installed: in this Python environment (the [sync] extra, e.g.
    inside a pipx environment whose scripts are not on PATH), or as a program on PATH."""
    if importlib.util.find_spec("ffsubsync") is not None:
        return [sys.executable, "-m", "ffsubsync.ffsubsync"]
    for name in ("ffs", "ffsubsync"):
        found = shutil.which(name)
        if found:
            return [found]
    return None


def resync(
    video: Path,
    stream_index: int,
    text: str,
    workdir: Path,
    *,
    ffmpeg: str,
    fps: float | None = None,
    language: str | None = None,
) -> str:
    """The subtitles in `text`, re-timed to the speech in audio stream `stream_index` of `video`."""
    command = ffsubsync_command()
    if command is None:
        raise SubtitleError("ffsubsync is not installed (pip install 'video-beep-remover[sync]')")
    source = workdir / "resync-in.srt"
    target = workdir / "resync-out.srt"
    save_subtitles(text, source, fps=fps)
    target.unlink(missing_ok=True)
    args = [
        *command, str(video), "-i", str(source), "-o", str(target),
        "--reference-stream", f"0:{stream_index}", "--ffmpeg-path", str(Path(ffmpeg).parent),
    ]  # fmt: skip
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SubtitleError(f"ffsubsync did not finish: {exc}") from exc
    if result.returncode != 0 or not target.is_file():
        lines = (result.stderr or result.stdout).strip().splitlines()
        raise SubtitleError(f"ffsubsync failed: {lines[-1] if lines else f'exit code {result.returncode}'}")
    return read_subtitle_file(target, language)
