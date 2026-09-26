"""Check that another audio stream carries the same dialogue as the analysed one (DESIGN.md §6.11).

With other_audio_streams = "auto", a stream in the analysed stream's language gets the same mutes,
which covers e.g. a stereo downmix next to the 5.1 mix. A mislabelled dub or a commentary track would
keep its words, and get mutes in the wrong places, so the streams are compared first: their 16 kHz
mono downmixes are cross-correlated around a few of the muted spans."""

import math
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from video_beep_remover.media.audio import SAMPLE_RATE, Audio
from video_beep_remover.models import CensorInterval

SPANS = 5  # muted spans compared, spread over the file
CONTEXT_S = 1.0  # audio compared on either side of each span
MAX_LAG_S = 0.1  # how far apart the streams may be (a downmix can add a small delay)
MIN_CORRELATION = 0.5  # to be tuned on the evaluation set (DESIGN.md §11)
SILENT_RMS = 10 ** (-60 / 20)  # windows this quiet in either stream prove nothing

Reader = Callable[[int, float, float], Audio]  # (stream index, start, end) -> 16 kHz mono samples


@dataclass(frozen=True)
class DialogueCheck:
    same: bool
    correlation: float | None  # median over the compared spans; None when there was nothing to compare
    lag: float | None  # seconds the other stream is behind the analysed one
    compared: int  # spans with sound in both streams

    def describe(self) -> str:
        if self.correlation is None:
            return "nothing to compare: silent around every muted span"
        lag = f", {self.lag * 1000:+.0f} ms apart" if self.lag else ""
        return f"correlation {self.correlation:.2f} over {self.compared} spans{lag}"


def correlation(a: Audio, b: Audio, max_lag: int) -> tuple[float, int]:
    """The highest normalized cross-correlation of two signals within ±max_lag samples, and its lag
    (positive: `b` is behind `a`)."""
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)
    norm = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if n == 0 or norm == 0.0:
        return 0.0, 0
    size = 1 << (2 * n - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    values = np.concatenate([full[: max_lag + 1], full[size - max_lag :]])
    best = int(np.argmax(np.abs(values)))
    return abs(float(values[best])) / norm, int(lags[best])


def _rms(samples: Audio) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0


def spread(intervals: Sequence[CensorInterval], count: int = SPANS) -> list[CensorInterval]:
    """Up to `count` intervals spread evenly over the list."""
    if len(intervals) <= count:
        return list(intervals)
    step = (len(intervals) - 1) / (count - 1)
    return [intervals[round(i * step)] for i in range(count)]


def same_dialogue(
    read: Reader,
    analysed: int,
    other: int,
    intervals: Sequence[CensorInterval],
    duration: float,
    *,
    workers: int = 4,
) -> DialogueCheck:
    """Whether stream `other` carries the dialogue of stream `analysed` around the muted spans."""
    windows = [(max(0.0, i.start - CONTEXT_S), min(duration, i.end + CONTEXT_S)) for i in spread(intervals)]
    jobs = [(stream, start, end) for start, end in windows for stream in (analysed, other)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        audio = list(pool.map(lambda job: read(*job), jobs))
    scores: list[tuple[float, int]] = []
    for number in range(len(windows)):
        a, b = audio[2 * number], audio[2 * number + 1]
        if _rms(a) < SILENT_RMS or _rms(b) < SILENT_RMS:
            continue
        scores.append(correlation(a, b, round(MAX_LAG_S * SAMPLE_RATE)))
    if not scores:
        return DialogueCheck(True, None, None, 0)  # muting silence is harmless
    median = statistics.median(score for score, _ in scores)
    lag = statistics.median_low(lag for _, lag in scores) / SAMPLE_RATE
    return DialogueCheck(median >= MIN_CORRELATION, median, lag, len(scores))
