import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import words
from video_beep_remover.config.schema import Category, Hints, LexiconConfig
from video_beep_remover.detect.confirm import (
    EDGE_S,
    assemble,
    attribute,
    dedupe,
    detect_in_windows,
    resolve_unconfirmed,
    split_confirmed,
)
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.detect.planner import (
    FlaggedCue,
    FlaggedWord,
    audio_seconds,
    flag_cues,
    flagged_windows,
    plan_windows,
    search_padding,
    uncovered_speech,
)
from video_beep_remover.models import Cue, Detection, SyncModel, Window

LEXICON = compile_lexicon(
    LexiconConfig(
        categories={"strong": Category(terms=["*shit*", "son of a bitch"]), "mild": Category(terms=["hell"])},
        hints=Hints(terms=["freak*", "heck"]),
    )
)


def plan(*spans: tuple[float, float], duration: float = 100.0, **kwargs: float) -> list[tuple[float, float]]:
    settings = {"min_window": 4.0, "max_window": 30.0, "merge_gap": 1.0} | kwargs
    windows = plan_windows([Window(s, e) for s, e in spans], duration=duration, **settings)
    return [(round(w.start, 6), round(w.end, 6)) for w in windows]


def test_cues_are_flagged_with_reasons_and_word_positions() -> None:
    cues = [
        Cue(1, 1, 3, "What the hell--no!"),
        Cue(2, 4, 6, "Get the f*** out, you son of a bitch."),
        Cue(3, 7, 9, "Freaking unbelievable."),
        Cue(4, 10, 12, "Nothing to see here."),
    ]
    flags = {flag.cue.index: flag for flag in flag_cues(LEXICON, cues)}
    assert sorted(flags) == [1, 2, 3]
    assert flags[1].reasons == {"lexicon"} and flags[1].strong
    assert flags[1].words == (FlaggedWord(9, 13, "hell", "mild"),)
    assert flags[2].reasons == {"lexicon", "masked"}
    assert [cues[1].text[w.start : w.end] for w in flags[2].words] == ["f***", "son of a bitch."]
    assert flags[3].reasons == {"hint"} and not flags[3].strong and flags[3].words == ()


def test_windows_grow_to_the_minimum_and_stay_inside_the_file() -> None:
    assert plan((10.0, 11.0)) == [(8.5, 12.5)]
    assert plan((-1.0, 0.5)) == [(0.0, 4.0)]
    assert plan((99.0, 101.0), duration=100.0) == [(96.0, 100.0)]
    assert plan((120.0, 130.0), duration=100.0) == []  # a cue past the end of a truncated file
    assert plan((1.0, 2.0), duration=3.0) == [(0.0, 3.0)]  # a file shorter than min_window


def test_close_windows_merge_and_long_ones_split_with_overlap() -> None:
    assert plan((10.0, 15.0), (15.9, 20.0)) == [(10.0, 20.0)]
    assert plan((10.0, 15.0), (16.0, 20.0)) == [(10.0, 15.0), (16.0, 20.0)]
    pieces = plan((10.0, 70.0))
    assert len(pieces) == 3 and pieces[0][0] == 10.0 and pieces[-1][1] == 70.0
    assert [round(end - start, 3) for start, end in pieces] == [21.333] * 3
    assert [round(a[1] - b[0], 3) for a, b in itertools.pairwise(pieces)] == [2.0, 2.0]


def test_split_pieces_keep_the_cues_they_touch() -> None:
    spans = [Window(10, 12, frozenset({"lexicon"}), (1,)), Window(12.5, 60, frozenset({"uncovered"}))]
    first, second = plan_windows(spans, duration=100, min_window=4, max_window=30, merge_gap=1)
    assert (first.reasons, first.cues) == ({"lexicon", "uncovered"}, (1,))
    assert (second.reasons, second.cues) == ({"uncovered"}, ())


@given(
    spans=st.lists(
        st.tuples(st.floats(-5, 105), st.floats(0, 40)).map(lambda t: (t[0], t[0] + t[1])), max_size=30
    ),
    min_window=st.floats(0, 10),
    max_window=st.floats(5, 30),
    merge_gap=st.floats(0, 3),
)
def test_planned_windows_are_bounded_and_cover_every_span(
    spans: list[tuple[float, float]], min_window: float, max_window: float, merge_gap: float
) -> None:
    windows = plan_windows(
        [Window(s, e) for s, e in spans],
        duration=100.0,
        min_window=min_window,
        max_window=max_window,
        merge_gap=merge_gap,
    )
    for window in windows:
        assert 0.0 <= window.start < window.end <= 100.0
        assert window.duration <= max_window + 1e-9
    for start, end in spans:
        start, end = max(0.0, start), min(100.0, end)
        if end > start:
            # every moment of the span is inside some window
            for point in (start, (start + end) / 2, end):
                assert any(w.start - 1e-9 <= point <= w.end + 1e-9 for w in windows), (start, end, point)


def test_uncovered_speech_is_what_no_cue_covers() -> None:
    cues = [Cue(1, 10.0, 12.0, "a"), Cue(2, 20.0, 25.0, "b")]
    speech = [(5.0, 11.0), (12.2, 12.6), (15.0, 30.0), (40.0, 40.3)]
    assert uncovered_speech(speech, cues, SyncModel()) == [(5.0, 9.5), (15.0, 19.5), (25.5, 30.0)]
    shifted = uncovered_speech([(0.0, 50.0)], cues, SyncModel(offset=2.0))
    assert shifted == [(0.0, 11.5), (14.5, 21.5), (27.5, 50.0)]


def test_search_padding_grows_with_the_sync_error() -> None:
    assert search_padding(1.5, SyncModel(error=0.1)) == pytest.approx(1.8)
    flag = FlaggedCue(Cue(3, 10.0, 12.0, "hell"), frozenset({"lexicon"}), ())
    [window] = flagged_windows([flag], SyncModel(offset=1.0), pad=1.5)
    assert (window.start, window.end, window.cues) == (9.5, 14.5, (3,))
    assert audio_seconds([window, Window(0, 2)]) == 7.0


def test_words_near_window_edges_are_dropped_unless_at_the_file_edges() -> None:
    windows = [Window(10.0, 20.0), Window(0.0, 5.0)]
    transcripts = [
        words(("early", 10.1, 10.2), ("kept", 12.0, 12.3), ("late", 19.8, 19.95)),
        words(("start", 0.05, 0.2), ("end", 4.6, 4.8)),
    ]
    heard = assemble(windows, transcripts, duration=5.0 + 100)
    assert [[w.text for w in group] for group in heard] == [["start"], ["kept"]]
    at_end = assemble([Window(90.0, 100.0)], [words(("last", 99.8, 99.95))], duration=100.0)
    assert [w.text for w in at_end[0]] == ["last"]
    trimmed = assemble(windows[:1], transcripts[:1], duration=100.0, clean_edges=[(True, False)])
    assert [w.text for w in trimmed[0]] == ["early", "kept"]  # a window trimmed to silence has a clean start
    assert EDGE_S == 0.3


def test_overlapping_pieces_split_their_words_at_the_middle_of_the_overlap() -> None:
    windows = [Window(10.0, 30.0), Window(28.0, 50.0)]
    first = words(("a", 20.0, 20.3), ("b", 28.5, 28.8), ("c", 29.2, 29.5))
    second = words(("b", 28.52, 28.8), ("c", 29.21, 29.5), ("d", 40.0, 40.2))
    [group] = assemble(windows, [first, second], duration=100.0)
    assert [(w.text, w.start) for w in group] == [("a", 20.0), ("b", 28.5), ("c", 29.21), ("d", 40.0)]


def test_phrases_never_span_separate_windows() -> None:
    windows = [Window(0.0, 10.0), Window(20.0, 30.0)]
    transcripts = [words(("son", 8, 8.2), ("of", 8.3, 8.4), ("a", 8.5, 8.6)), words(("bitch", 21, 21.3))]
    detections, heard = detect_in_windows(LEXICON, windows, transcripts, duration=100.0)
    assert detections == [] and heard == 4


def detection(start: float, end: float, term: str = "hell", confidence: float = 0.9) -> Detection:
    return Detection(start, end, term, term, "mild", confidence)


def test_dedupe_keeps_the_most_confident_repeat() -> None:
    kept = dedupe(
        [
            detection(5.0, 5.3, confidence=0.5),
            detection(5.1, 5.35, confidence=0.8),
            detection(5.1, 5.3, "shit"),
        ]
    )
    assert sorted((d.term, d.confidence) for d in kept) == [("hell", 0.8), ("shit", 0.9)]
    assert len(dedupe([detection(0.0, 10.0), detection(3.0, 3.2, "shit"), detection(5.0, 5.3)])) == 2


def test_confirmation_and_unconfirmed_resolutions() -> None:
    cue = Cue(7, 10.0, 12.0, "Get the f*** out!")
    flag = FlaggedCue(cue, frozenset({"masked"}), (FlaggedWord(8, 12, "f***", "masked"),))
    hint = FlaggedCue(Cue(8, 30.0, 31.0, "heck"), frozenset({"hint"}), ())
    sync = SyncModel(offset=1.0)
    confirmed, unconfirmed = split_confirmed([flag, hint], [detection(12.2, 12.5)], sync, pad=0.0)
    assert (confirmed, unconfirmed) == ([flag], [])
    confirmed, unconfirmed = split_confirmed([flag, hint], [detection(20.0, 20.5)], sync, pad=1.5)
    assert (confirmed, unconfirmed) == ([], [flag])

    [estimate] = resolve_unconfirmed(flag, "estimate", sync, duration=100.0)
    # characters 8-12 of 17, spread over 11.0-13.0 s, widened by 0.3 s
    assert (estimate.start, estimate.end) == (
        pytest.approx(11.0 + 2 * 8 / 17 - 0.3),
        pytest.approx(11.0 + 2 * 12 / 17 + 0.3),
    )
    assert (estimate.heard, estimate.source, estimate.cue, estimate.confidence) == (
        "f***",
        "estimate",
        7,
        0.2,
    )
    [whole] = resolve_unconfirmed(flag, "cue", sync, duration=100.0)
    assert (whole.start, whole.end, whole.source) == (11.0, 13.0, "cue")
    assert resolve_unconfirmed(flag, "skip", sync, duration=100.0) == []


def test_detections_are_attributed_to_the_cue_they_confirm() -> None:
    flag = FlaggedCue(Cue(7, 10.0, 12.0, "hell"), frozenset({"lexicon"}), ())
    near, far = attribute([detection(12.5, 12.8), detection(40.0, 40.2)], [flag], SyncModel(), pad=1.0)
    assert (near.cue, far.cue) == (7, None)
