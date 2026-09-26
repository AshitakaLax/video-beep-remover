"""Run ffmpeg and ffprobe as subprocesses (argument lists only, never a shell)."""

import json
import logging
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from video_beep_remover.errors import DependencyError, MediaError

log = logging.getLogger(__name__)

MIN_VERSION = (5, 1)
_VERSION = re.compile(r"version\s+n?(\d+)\.(\d+)")


def file_arg(path: Path) -> str:
    """Absolute path with the file: protocol, safe for names that start with '-' or contain ':'."""
    return "file:" + str(path.resolve())


@dataclass(frozen=True)
class FFmpegVersion:
    text: str  # first line of `ffmpeg -version`
    major: int | None
    minor: int | None

    @classmethod
    def parse(cls, banner: str) -> "FFmpegVersion":
        first = banner.strip().splitlines()[0] if banner.strip() else ""
        match = _VERSION.search(first)
        if match:
            return cls(first, int(match.group(1)), int(match.group(2)))
        return cls(first, None, None)  # git builds ("N-112345-g...") report no release number

    @property
    def number(self) -> str:
        """The release, e.g. "6.1.1-3ubuntu5" (or the whole first line if there is none)."""
        match = re.search(r"version\s+(\S+)", self.text)
        return match.group(1) if match else self.text

    def at_least(self, major: int, minor: int = 0) -> bool:
        """Unknown versions (git builds) are assumed to be recent."""
        if self.major is None or self.minor is None:
            return True
        return (self.major, self.minor) >= (major, minor)


class FFmpeg:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        found_ffmpeg = shutil.which(ffmpeg)
        found_ffprobe = shutil.which(ffprobe)
        if not found_ffmpeg or not found_ffprobe:
            missing = ", ".join(
                name for name, found in ((ffmpeg, found_ffmpeg), (ffprobe, found_ffprobe)) if not found
            )
            raise DependencyError(
                f"not found: {missing}. Install FFmpeg 5.1 or later and make sure ffmpeg and ffprobe are on "
                "PATH, or set [tools] ffmpeg/ffprobe in the config."
            )
        self.ffmpeg = found_ffmpeg
        self.ffprobe = found_ffprobe

    @cached_property
    def version(self) -> FFmpegVersion:
        result = subprocess.run([self.ffmpeg, "-version"], capture_output=True, text=True, check=False)
        return FFmpegVersion.parse(result.stdout)

    @cached_property
    def encoders(self) -> frozenset[str]:
        result = subprocess.run(
            [self.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
        )
        names = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            # Encoder lines look like " A....D aac                  AAC (Advanced Audio Coding)".
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS" and parts[1] != "=":
                names.add(parts[1])
        return frozenset(names)

    def filter_script_args(self, script_name: str) -> list[str]:
        """-/filter_complex FILE on FFmpeg 7+, where -filter_complex_script is deprecated."""
        if self.version.at_least(7):
            return ["-/filter_complex", script_name]
        return ["-filter_complex_script", script_name]

    def probe(self, path: Path) -> dict[str, Any]:
        args = [
            self.ffprobe, "-v", "error", "-show_format", "-show_streams", "-show_chapters",
            "-of", "json", file_arg(path),
        ]  # fmt: skip
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise MediaError(f"ffprobe could not read {path}: {result.stderr.strip() or 'unknown error'}")
        data: dict[str, Any] = json.loads(result.stdout or "{}")
        return data

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        """Run ffmpeg; `on_progress` receives the output position in seconds."""
        command = [self.ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
        if on_progress is not None:
            command += ["-progress", "pipe:1", "-nostats"]
        command += list(args)
        log.debug("running: %s", subprocess.list2cmdline(command))
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE if on_progress else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        stderr_tail: deque[str] = deque(maxlen=40)
        assert process.stderr is not None
        reader = threading.Thread(target=lambda: stderr_tail.extend(process.stderr or ()), daemon=True)
        reader.start()
        if on_progress is not None and process.stdout is not None:
            for line in process.stdout:
                if line.startswith("out_time_us="):
                    value = line.split("=", 1)[1].strip()
                    if value.lstrip("-").isdigit():
                        on_progress(max(0, int(value)) / 1_000_000)
        code = process.wait()
        reader.join(timeout=5)
        if code != 0:
            detail = "".join(stderr_tail).strip() or "no error output"
            raise MediaError(f"ffmpeg exited with code {code}: {detail}")

    def capture(self, args: Sequence[str]) -> bytes:
        """Run ffmpeg writing to pipe:1 and return the output bytes."""
        command = [self.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *args]
        log.debug("running: %s", subprocess.list2cmdline(command))
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip() or "no error output"
            raise MediaError(f"ffmpeg exited with code {result.returncode}: {detail}")
        return result.stdout
