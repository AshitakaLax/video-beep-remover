import numpy as np
import pytest

from video_beep_remover.detect.refine import SEARCH_S, refine_edges
from video_beep_remover.media.audio import SAMPLE_RATE, ArrayAudioSource, Audio
from video_beep_remover.models import CensorInterval


def speech(duration: float, *pauses: tuple[float, float]) -> Audio:
    """Loud noise standing in for speech, silent in the pauses."""
    audio = np.random.default_rng(1).uniform(-0.5, 0.5, round(duration * SAMPLE_RATE)).astype(np.float32)
    for start, end in pauses:
        audio[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)] = 0.0
    return audio


def refine(audio: Audio, *spans: tuple[float, float], merge_gap: float = 0.1) -> list[tuple[float, float]]:
    intervals = [CensorInterval(start, end) for start, end in spans]
    refined = refine_edges(
        intervals, ArrayAudioSource(audio), duration=len(audio) / SAMPLE_RATE, merge_gap=merge_gap
    )
    return [(round(i.start, 3), round(i.end, 3)) for i in refined]


def test_edges_in_speech_move_out_to_the_nearest_pause() -> None:
    audio = speech(3.0, (1.00, 1.03), (2.50, 2.52))
    # The fade out then runs through the pause's last 10 ms, the fade in through its first.
    assert refine(audio, (1.07, 2.45)) == [(1.02, 2.51)]


def test_edges_already_in_a_pause_stay() -> None:
    audio = speech(3.0, (0.9, 1.2), (2.3, 2.6))
    assert refine(audio, (1.1, 2.4)) == [(1.1, 2.4)]


def tone(duration: float, *pauses: tuple[float, float]) -> Audio:
    """A 100 Hz tone: every 10 ms frame holds one whole cycle, so every frame is equally loud."""
    t = np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    for start, end in pauses:
        audio[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)] = 0.0
    return audio


def test_edges_move_outward_and_at_most_the_search_distance() -> None:
    assert refine(tone(3.0), (1.0, 2.0)) == [(1.0, 2.0)]  # nothing quieter nearby: nothing moves
    assert SEARCH_S == 0.08
    in_reach = tone(3.0, (0.92, 0.93), (2.07, 2.08))
    assert refine(in_reach, (1.0, 2.0)) == [(0.92, 2.08)]
    # Pauses 90 ms outside the edges are out of reach, and the one inside never pulls an edge in.
    out_of_reach = tone(3.0, (0.90, 0.91), (1.5, 1.6), (2.09, 2.10))
    assert refine(out_of_reach, (1.0, 2.0)) == [(1.0, 2.0)]


def test_refined_intervals_stay_in_the_file_and_are_merged_again() -> None:
    audio = speech(2.0, (0.0, 0.02), (0.95, 0.97), (1.97, 2.0))
    assert refine(audio, (0.03, 0.9), (1.02, 1.95), merge_gap=0.1) == [(0.01, 1.98)]
    assert refine(audio, (0.03, 0.9), (1.02, 1.95), merge_gap=0.0) == [(0.01, 0.96), (0.96, 1.98)]


class Reads:
    def __init__(self, audio: Audio) -> None:
        self.audio = ArrayAudioSource(audio)
        self.spans: list[tuple[float, float]] = []

    def read(self, start: float, end: float) -> Audio:
        self.spans.append((round(start, 3), round(end, 3)))
        return self.audio.read(start, end)


def test_nearby_intervals_are_read_from_the_file_together() -> None:
    source = Reads(speech(30.0))
    intervals = [CensorInterval(1.0, 2.0), CensorInterval(4.0, 5.0), CensorInterval(20.0, 21.0)]
    refine_edges(intervals, source, duration=30.0, merge_gap=0.25)
    assert source.spans == [(0.91, 5.09), (19.91, 21.09)]


@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_an_interval_covering_the_whole_file_is_unchanged(duration: float) -> None:
    assert refine(speech(duration), (0.0, duration)) == [(0.0, duration)]
