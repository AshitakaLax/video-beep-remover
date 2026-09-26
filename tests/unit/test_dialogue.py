import numpy as np
import pytest

from video_beep_remover.media.audio import SAMPLE_RATE
from video_beep_remover.media.dialogue import MAX_LAG_S, correlation, same_dialogue, spread
from video_beep_remover.models import CensorInterval as Span

RNG = np.random.default_rng(7)
SPEECH = RNG.standard_normal(60 * SAMPLE_RATE).astype(np.float32)  # "the dialogue", 60 s


def reader(tracks: dict[int, np.ndarray]):  # type: ignore[no-untyped-def]
    def read(index: int, start: float, end: float) -> np.ndarray:
        return tracks[index][round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)]

    return read


def test_correlation_finds_the_lag() -> None:
    delayed = np.concatenate([np.zeros(480, np.float32), SPEECH[:-480]])
    score, lag = correlation(SPEECH[:40000], delayed[:40000], round(MAX_LAG_S * SAMPLE_RATE))
    assert score > 0.95 and lag == 480
    assert correlation(SPEECH[:40000], np.zeros(40000, np.float32), 100) == (0.0, 0)


def test_spread_picks_evenly() -> None:
    spans = [Span(i, i + 0.5) for i in range(10)]
    assert [s.start for s in spread(spans, 5)] == [0, 2, 4, 7, 9]  # 9/4 steps, rounded
    assert spread(spans[:3], 5) == spans[:3]


SPANS = [Span(5.0, 5.5), Span(20.0, 20.4), Span(41.0, 41.3)]


def test_a_downmix_of_the_same_mix_matches() -> None:
    music = RNG.standard_normal(SPEECH.size).astype(np.float32)
    mix = SPEECH + 0.5 * music
    downmix = 0.7 * SPEECH + 0.3 * music  # different weights, same content
    check = same_dialogue(reader({1: mix, 2: downmix}), 1, 2, SPANS, 60.0)
    assert check.same and check.correlation is not None and check.correlation > 0.9 and check.compared == 3


def test_other_dialogue_over_the_same_music_does_not() -> None:
    music = RNG.standard_normal(SPEECH.size).astype(np.float32)
    dub = RNG.standard_normal(SPEECH.size).astype(np.float32)  # another language's dialogue
    check = same_dialogue(reader({1: SPEECH + 0.3 * music, 2: dub + 0.3 * music}), 1, 2, SPANS, 60.0)
    assert not check.same
    assert "correlation 0.0" in check.describe()


def test_a_stream_offset_beyond_the_allowed_lag_does_not_match() -> None:
    late = np.concatenate([np.zeros(SAMPLE_RATE // 2, np.float32), SPEECH[: -SAMPLE_RATE // 2]])
    assert not same_dialogue(reader({1: SPEECH, 2: late}), 1, 2, SPANS, 60.0).same
    near = np.concatenate([np.zeros(SAMPLE_RATE // 50, np.float32), SPEECH[: -SAMPLE_RATE // 50]])
    check = same_dialogue(reader({1: SPEECH, 2: near}), 1, 2, SPANS, 60.0)
    assert check.same and check.lag == pytest.approx(0.02)


def test_silence_proves_nothing() -> None:
    silent = np.zeros(SPEECH.size, np.float32)
    check = same_dialogue(reader({1: SPEECH, 2: silent}), 1, 2, SPANS, 60.0)
    assert (check.same, check.correlation, check.compared) == (True, None, 0)
