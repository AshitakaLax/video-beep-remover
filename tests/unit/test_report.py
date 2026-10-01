from pathlib import Path

import pytest

from video_beep_remover.config import load_config
from video_beep_remover.config.loader import config_hash
from video_beep_remover.errors import UsageError
from video_beep_remover.models import CensorInterval as Span
from video_beep_remover.models import Detection
from video_beep_remover.report import report_detections, report_intervals, review_srt, srt_time, write_text


def test_review_srt_names_what_was_heard_in_each_span() -> None:
    detections = [
        Detection(1.0, 1.3, " hell", "hell", "mild", 0.9),
        Detection(1.35, 1.6, "damn", "damn", "mild", 0.8),
        Detection(5.0, 5.4, "f***", "*fuck*", "strong", 0.2, "estimate", 4),
    ]
    text = review_srt([Span(0.88, 1.72), Span(4.7, 5.7), Span(9.0, 9.5)], detections, shift=0.021)
    assert text == (
        "1\n00:00:00,901 --> 00:00:01,741\n[muted] hell, damn\n\n"
        "2\n00:00:04,721 --> 00:00:05,721\n[muted] f*** (estimated from subtitles)\n\n"
        "3\n00:00:09,021 --> 00:00:09,521\n[muted]\n"
    )
    assert srt_time(3725.5) == "01:02:05,500" and srt_time(-0.2) == "00:00:00,000"


def test_a_failed_write_leaves_the_file_as_it_was(tmp_path: Path) -> None:
    edl = tmp_path / "movie.edl"
    write_text(edl, "1.000\t2.000\t1\n")
    with pytest.raises(UnicodeEncodeError):
        write_text(edl, "3.000\t4.000\t1\n\ud800")  # fails partway through
    assert edl.read_text("utf-8") == "1.000\t2.000\t1\n"


def test_report_intervals_are_sorted_merged_and_checked() -> None:
    data = {"intervals": [{"start": 5, "end": 6}, {"start": "1.5", "end": 2}, {"start": 1.8, "end": 3}]}
    assert report_intervals(data) == [Span(1.5, 3.0), Span(5.0, 6.0)]
    assert report_intervals({"intervals": [{"start": -1, "end": 0.5}]}) == [Span(0.0, 0.5)]
    for bad, message in [
        ({}, 'no "intervals" list'),
        ({"intervals": [{"start": 1}]}, r"intervals\[0\]: needs a numeric"),
        ({"intervals": [{"start": "x", "end": 2}]}, r"intervals\[0\]: needs a numeric"),
        (
            {"intervals": [{"start": 1, "end": 2}, {"start": 4, "end": 3}]},
            r"intervals\[1\]: end must be after",
        ),
        ({"intervals": [{"start": 1, "end": float("nan")}]}, "end must be after"),
    ]:
        with pytest.raises(UsageError, match=message):
            report_intervals(bad)


def test_report_detections_skip_what_they_cannot_read() -> None:
    data = {
        "detections": [
            {"start": 1.0, "end": 1.3, "heard": "hell", "term": "hell", "category": "mild", "confidence": 0.9,
             "source": "asr", "cue": 3},
            {"start": 2.0, "heard": "no end"},
            "not even a dict",
            {"start": 4.0, "end": 4.2, "source": "made up"},
        ]
    }  # fmt: skip
    found = report_detections(data)
    assert [(d.heard, d.source, d.cue) for d in found] == [("hell", "asr", 3), ("", "asr", None)]


def test_config_hash_ignores_secrets_but_not_settings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def hashed(**overrides: object) -> str:
        return config_hash(load_config(None, env={}, cwd=tmp_path, overrides=dict(overrides)).config)

    base = hashed(**{"subtitles.opensubtitles.api_key": "one"})
    assert base == hashed(**{"subtitles.opensubtitles.api_key": "two"})
    assert base != hashed(**{"subtitles.opensubtitles.api_key": "one", "censor.pad_after_ms": 250})
    assert len(base) == 12
