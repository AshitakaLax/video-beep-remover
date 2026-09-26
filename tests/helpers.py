"""Helpers for building test media and measuring what the renderer did."""

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_beep_remover.models import Word

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
SR = 48_000
TONE_AMPLITUDE = 0.125  # the lavfi sine source's peak level


def run_ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@dataclass
class Track:
    frequency: int = 440
    language: str | None = "eng"
    title: str | None = None
    default: bool = False
    delay: float = 0.0


def make_clip(
    path: Path,
    *,
    duration: float = 6.0,
    tracks: Sequence[Track] = (Track(default=True),),
    audio_codec: str = "aac",
    subtitles: str | None = None,
    video: bool = True,
) -> Path:
    """A test-pattern video with one pure tone per audio track (tones stand in for speech)."""
    args: list[str] = []
    maps: list[str] = []
    index = 0
    if video:
        args += ["-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=24:duration={duration}"]
        maps += ["-map", f"{index}:v"]
        index += 1
    for track in tracks:
        if track.delay:
            args += ["-itsoffset", str(track.delay)]
        args += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={track.frequency}:sample_rate={SR}:duration={duration}",
        ]
        maps += ["-map", f"{index}:a"]
        index += 1
    if subtitles is not None:
        srt = path.with_suffix(".srt")
        srt.write_text(subtitles, "utf-8")
        args += ["-i", str(srt)]
        maps += ["-map", f"{index}:s"]
    meta: list[str] = []
    for position, track in enumerate(tracks):
        if track.language:
            meta += [f"-metadata:s:a:{position}", f"language={track.language}"]
        if track.title:
            meta += [f"-metadata:s:a:{position}", f"title={track.title}"]
        meta += [f"-disposition:a:{position}", "default" if track.default else "0"]
    codecs = ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", audio_codec]
    if subtitles is not None:
        codecs += ["-c:s", "mov_text" if path.suffix == ".mp4" else "srt"]
    run_ffmpeg(*args, *maps, *codecs, *meta, str(path))
    return path


def decode(path: Path, stream: str = "0:a:0") -> np.ndarray:
    """Mono float32 at 48 kHz; sample 0 is the start of the file."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", stream,
         "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
        capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(result.stdout, dtype=np.float32)


def tone_gain(samples: np.ndarray, at: float, frequency: int = 440, window: float = 0.004) -> float:
    """Level of `frequency` around time `at`, relative to the unmuted tone (1.0 = untouched)."""
    start = max(0, int((at - window / 2) * SR))
    block = samples[start : start + int(window * SR)]
    n = np.arange(block.size)
    magnitude = np.abs(np.dot(block, np.exp(-2j * np.pi * frequency * n / SR))) * 2 / max(1, block.size)
    return float(magnitude / TONE_AMPLITUDE)


class FakeTranscriber:
    """Returns scripted words instead of running Whisper."""

    name = "fake"

    def __init__(self, words: Sequence[Word]) -> None:
        self.words = list(words)
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        offset: float,
        language: str,
        prompt: str | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[Word]:
        self.calls.append({"seconds": len(audio) / 16_000, "language": language, "prompt": prompt})
        if on_progress:
            on_progress(len(audio) / 16_000)
        return [Word(w.text, w.start + offset, w.end + offset, w.probability) for w in self.words]


def words(*items: tuple[str, float, float]) -> list[Word]:
    return [Word(text, start, end, 0.9) for text, start, end in items]
