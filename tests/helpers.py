"""Helpers for building test media and measuring what the renderer did."""

import contextlib
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_beep_remover.asr.base import Clip
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
    sample_rate: int = SR,
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
            f"sine=frequency={track.frequency}:sample_rate={sample_rate}:duration={duration}",
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
    """Returns scripted words instead of running Whisper. Words are in media time; each clip gets the
    words that lie inside it."""

    name = "fake"

    def __init__(self, words: Sequence[Word]) -> None:
        self.words = list(words)
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        clips: Sequence[Clip],
        *,
        language: str,
        prompt: str | None,
        vad: bool = False,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[list[Word]]:
        results = []
        done = 0.0
        for clip in clips:
            seconds = len(clip.audio) / 16_000
            end = clip.start + seconds
            self.calls.append(
                {"start": clip.start, "seconds": seconds, "language": language, "prompt": prompt, "vad": vad}
            )
            results.append([w for w in self.words if clip.start - 1e-6 <= w.start and w.end <= end + 1e-6])
            done += seconds
            if on_progress:
                on_progress(done)
        return results


def words(*items: tuple[str, float, float]) -> list[Word]:
    return [Word(text, start, end, 0.9) for text, start, end in items]


def say(text: str, start: float, end: float, probability: float = 0.9) -> list[Word]:
    """The words of `text` spread evenly over [start, end], as Whisper would report them."""
    parts = text.split()
    step = (end - start) / len(parts)
    return [
        Word(" " + part, start + i * step, start + (i + 1) * step - 0.02, probability)
        for i, part in enumerate(parts)
    ]


def srt(*cues: tuple[float, float, str]) -> str:
    """SubRip text for (start, end, text) cues."""

    def stamp(t: float) -> str:
        ms = round(t * 1000)
        return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "".join(f"{i}\n{stamp(s)} --> {stamp(e)}\n{text}\n\n" for i, (s, e, text) in enumerate(cues, 1))


class StrictUI:
    """A UI that fails the test when progress displays nest: Rich 13 raises LiveError for that."""

    def __init__(self) -> None:
        self.active: str | None = None

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    @contextlib.contextmanager
    def _live(self, label: str) -> Iterator[None]:
        assert self.active is None, f"{label!r} started inside {self.active!r}"
        self.active = label
        try:
            yield
        finally:
            self.active = None

    @contextlib.contextmanager
    def progress(self, label: str, total: float) -> Iterator[Callable[[float], None]]:
        with self._live(label):
            yield lambda done: None

    @contextlib.contextmanager
    def status(self, label: str) -> Iterator[None]:
        with self._live(label):
            yield
