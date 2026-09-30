"""Decode audio for speech recognition: 16 kHz mono float32 on the media timeline (DESIGN.md §6.2)."""

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import StreamInfo

SAMPLE_RATE = 16_000
# Pads a late-starting track with silence so that sample 0 is exactly the requested time.
_ALIGN = "aresample=async=1:first_pts=0"

Audio = npt.NDArray[np.float32]


def _decode_args(path: Path, stream_index: int, start: float | None, duration: float | None) -> list[str]:
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{max(0.0, start):.3f}"]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    return [
        *args, "-i", file_arg(path), "-map", f"0:{stream_index}", "-af", _ALIGN,
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le",
    ]  # fmt: skip


def read_window(ff: FFmpeg, path: Path, stream_index: int, start: float, duration: float) -> Audio:
    """Decode a short window into memory; sample 0 is `start`."""
    data = ff.capture([*_decode_args(path, stream_index, start, duration), "pipe:1"])
    return np.frombuffer(data, dtype=np.float32).copy()


def read_pcm_windows(
    ff: FFmpeg, path: Path, stream: StreamInfo, spans: Sequence[tuple[float, float]]
) -> list[npt.NDArray[np.float32]]:
    """Windows of an audio stream (media time) at its own rate, every channel kept: channels × samples.
    They are cut by sample count in one pass from the stream's start, stopping after the last one:
    seeking lands a few samples off in containers with coarse timestamps such as Matroska, and voice
    replacement subtracts the old voice sample-exactly (DESIGN.md §16). The renderer counts samples the
    same way.

    A window is cut short where the stream ended or FFmpeg failed before filling it: silence in its
    place would be subtracted from a voice that is still there."""
    rate, channels = stream.sample_rate or 48_000, stream.channels or 1
    offset = stream.start_time or 0.0
    ranges = [(max(0, round((a - offset) * rate)), max(0, round((b - offset) * rate))) for a, b in spans]
    windows = [np.zeros((channels, max(0, b - a)), dtype=np.float32) for a, b in ranges]
    last = max((b for _, b in ranges), default=0)
    command = [
        ff.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", file_arg(path),
        "-map", f"0:{stream.index}", "-f", "f32le", "pipe:1",
    ]  # fmt: skip
    frame = 4 * channels
    position = 0  # samples read so far
    pending = b""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        assert process.stdout is not None
        try:
            while position < last:
                data = pending + process.stdout.read(frame * rate)
                if len(data) < frame:
                    break
                whole = len(data) // frame * frame
                block = np.frombuffer(data[:whole], dtype=np.float32).reshape(-1, channels).T
                pending = data[whole:]
                for (a, b), window in zip(ranges, windows, strict=True):
                    first, stop = max(a, position), min(b, position + block.shape[1])
                    if first < stop:
                        window[:, first - a : stop - a] = block[:, first - position : stop - position]
                position += block.shape[1]
        finally:
            process.kill()  # the rest of the track is not needed
    return [window[:, : max(0, min(b, position) - a)] for (a, b), window in zip(ranges, windows, strict=True)]


def decode_track(
    ff: FFmpeg,
    path: Path,
    stream_index: int,
    dest: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Audio:
    """Decode a whole track to `dest` and memory-map it; sample 0 is the start of the file."""
    ff.run([*_decode_args(path, stream_index, None, None), file_arg(dest)], on_progress=on_progress)
    if dest.stat().st_size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.memmap(dest, dtype=np.float32, mode="r")


class AudioSource(Protocol):
    def read(self, start: float, end: float) -> Audio:
        """Samples from `start` to `end` (media seconds); sample 0 is `start`."""
        ...


class SeekingAudioSource:
    """Decodes each window straight from the file with input seeking (targeted mode)."""

    def __init__(self, ff: FFmpeg, path: Path, stream_index: int) -> None:
        self.ff = ff
        self.path = path
        self.stream_index = stream_index

    def read(self, start: float, end: float) -> Audio:
        return read_window(self.ff, self.path, self.stream_index, start, max(0.0, end - start))


class ArrayAudioSource:
    """Slices an already decoded track (hybrid mode decodes the whole track anyway)."""

    def __init__(self, audio: Audio) -> None:
        self.audio = audio

    def read(self, start: float, end: float) -> Audio:
        first = max(0, round(start * SAMPLE_RATE))
        return np.array(self.audio[first : max(first, round(end * SAMPLE_RATE))], dtype=np.float32)
