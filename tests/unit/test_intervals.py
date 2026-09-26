import itertools

from hypothesis import given
from hypothesis import strategies as st

from video_beep_remover.detect.intervals import build_intervals
from video_beep_remover.models import CensorInterval, Detection


def detection(start: float, end: float) -> Detection:
    return Detection(start=start, end=end, heard="x", term="x", category="c", confidence=1.0)


def build(*spans: tuple[float, float], duration: float = 100.0, **kwargs: float) -> list[CensorInterval]:
    settings = {"pad_before": 0.1, "pad_after": 0.1, "min_duration": 0.25, "merge_gap": 0.25} | kwargs
    return build_intervals([detection(s, e) for s, e in spans], duration=duration, **settings)


def test_pads_each_detection() -> None:
    assert build((10.0, 10.5)) == [CensorInterval(9.9, 10.6)]


def test_short_detections_grow_to_min_duration_around_their_middle() -> None:
    [interval] = build((10.0, 10.02), pad_before=0.0, pad_after=0.0, min_duration=0.3)
    assert interval.start == 10.01 - 0.15
    assert round(interval.duration, 9) == 0.3


def test_spans_at_file_edges_keep_min_duration() -> None:
    [start] = build((0.0, 0.01), pad_before=0.0, pad_after=0.0, min_duration=0.3)
    assert start.start == 0.0 and round(start.end, 9) == 0.3
    [end] = build((9.99, 10.0), duration=10.0, pad_before=0.0, pad_after=0.0, min_duration=0.3)
    assert end.end == 10.0 and round(end.start, 9) == 9.7


def test_merges_spans_closer_than_merge_gap() -> None:
    settings = {"pad_before": 0.0, "pad_after": 0.0, "min_duration": 0.0}
    assert build((1.0, 1.5), (1.8, 2.0), merge_gap=0.31, **settings) == [CensorInterval(1.0, 2.0)]
    assert len(build((1.0, 1.5), (1.8, 2.0), merge_gap=0.29, **settings)) == 2


def test_zero_length_detection_still_mutes_something() -> None:
    [interval] = build((5.0, 5.0), pad_before=0.0, pad_after=0.0, min_duration=0.0)
    assert interval.start < 5.0 < interval.end


def test_output_is_sorted() -> None:
    assert [i.start for i in build((50.0, 50.5), (10.0, 10.5))] == [9.9, 49.9]


def test_no_detections_no_intervals() -> None:
    assert build() == []


spans = st.lists(
    st.tuples(st.floats(0, 99), st.floats(0, 2)).map(lambda t: (t[0], min(100.0, t[0] + t[1]))), max_size=40
)


@given(
    spans=spans,
    pad=st.floats(0, 0.5),
    min_duration=st.floats(0, 1),
    merge_gap=st.floats(0, 1),
)
def test_intervals_are_sorted_disjoint_in_bounds_and_cover_every_detection(
    spans: list[tuple[float, float]], pad: float, min_duration: float, merge_gap: float
) -> None:
    result = build(*spans, pad_before=pad, pad_after=pad, min_duration=min_duration, merge_gap=merge_gap)
    for interval in result:
        assert 0.0 <= interval.start < interval.end <= 100.0
    for left, right in itertools.pairwise(result):
        assert right.start - left.end >= merge_gap
    for start, end in spans:
        assert any(i.start <= start + 1e-9 and end <= i.end + 1e-9 for i in result), (start, end)
