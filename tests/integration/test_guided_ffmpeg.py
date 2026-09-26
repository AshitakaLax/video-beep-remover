"""Subtitle-guided strategies end to end: real FFmpeg, scripted speech recognition."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers import FakeTranscriber, StrictUI, decode, extract_subtitles, make_clip, say, srt, tone_gain
from video_beep_remover.config import load_config
from video_beep_remover.errors import VbrError
from video_beep_remover.media.audio import SAMPLE_RATE
from video_beep_remover.models import Word
from video_beep_remover.pipeline import FileResult, Pipeline, RunOptions

pytestmark = pytest.mark.ffmpeg

DURATION = 60.0
LEAD = 0.2  # speech starts this long after its cue
LINES = [
    (2.0, 5.0, "We found the copper lantern near the harbor"),
    (10.0, 13.0, "What the hell is going on over there"),
    (20.0, 23.0, "Every pilgrim needs a marble saddle today"),
    (30.0, 33.0, "Get the f*** out of my orchard now"),
    (40.0, 43.0, "The glacier violin sounds like thunder again"),
    (46.0, 48.0, "Freaking unbelievable weather in this valley"),
    (50.0, 53.0, "Meadow falcons never sleep before dawn here"),
]
SUBTITLES = srt(*LINES)


def spoken(lines: list[tuple[float, float, str]] = LINES, **replace: str) -> list[Word]:
    """Words as heard: every line on time, with the masked word said in full unless replaced."""
    heard = []
    for start, end, text in lines:
        text = replace.get(text, text).replace("f***", "fuck")
        heard += say(text, start + LEAD, end - 0.3)
    return heard


class Run:
    def __init__(
        self,
        tmp_path: Path,
        *,
        words: list[Word],
        extra: list[Word] | None = None,
        speech: list[tuple[float, float]] | None = None,
        **overrides: Any,
    ) -> None:
        loaded = load_config(
            None, env={}, cwd=tmp_path, overrides={"transcription.device": "cpu", **overrides}
        )
        self.anchor = FakeTranscriber(words)
        self.main = FakeTranscriber(words + (extra or []))
        self.loaded: list[str] = []  # models loaded
        self.vad_runs = 0  # over the whole track

        def detect(
            audio: np.ndarray, on_progress: Callable[[float], None] | None = None
        ) -> list[tuple[float, float]]:
            whole_track = len(audio) >= (DURATION - 1) * SAMPLE_RATE
            self.vad_runs += whole_track
            return list(speech or []) if whole_track else []

        def load(choice: Any) -> FakeTranscriber:
            self.loaded.append(choice.name)
            return self.anchor if choice.name == "base.en" else self.main

        self.pipeline = Pipeline(loaded, ui=StrictUI(), transcriber_factory=load, speech_detector=detect)

    def process(self, source: Path, **options: Any) -> tuple[FileResult, dict[str, Any]]:
        result = self.pipeline.process(source, RunOptions(**({"dry_run": True} | options)))
        assert result.report is not None
        return result, json.loads(result.report.read_text("utf-8"))


def clip(tmp_path: Path, *, sidecar: str | None = SUBTITLES, embedded: str | None = None) -> Path:
    source = make_clip(tmp_path / "movie.mkv", duration=DURATION, subtitles=embedded)
    source.with_suffix(".srt").unlink(missing_ok=True)  # make_clip's temporary input for embedding
    if sidecar is not None:
        (tmp_path / "movie.en.srt").write_text(sidecar, "utf-8")
    return source


def test_targeted_transcribes_only_windows_around_flagged_cues(tmp_path: Path) -> None:
    source = clip(tmp_path)
    run = Run(tmp_path, words=spoken(), **{"analysis.strategy": "targeted"})
    result, report = run.process(source, dry_run=False)
    assert (result.status, result.strategy) == ("cleaned", "targeted")

    assert report["strategy"] == {"requested": "targeted", "used": "targeted", "fallback_reason": None}
    subtitle = report["subtitle"]
    assert (subtitle["source"], subtitle["label"], subtitle["cues"]) == ("sidecar", "movie.en.srt", 7)
    assert subtitle["sync"]["matched"] == subtitle["sync"]["anchors"] >= 5
    assert subtitle["sync"]["offset"] == pytest.approx(LEAD, abs=0.02)
    assert report["windows"]["flagged_cues"] == {"lexicon": 1, "masked": 1, "hint": 1}
    assert report["windows"]["count"] == 3
    assert report["confirmation"] == {"strong_flags": 2, "confirmed": 2}
    assert report["unconfirmed"] == []
    assert [(d["heard"], d["cue"], d["source"]) for d in report["detections"]] == [
        ("hell", 2, "asr"),
        ("fuck", 4, "asr"),
    ]

    # Only a few seconds around each flagged cue went through the main model, never the whole film.
    assert all(call["seconds"] < 10 and call["prompt"] for call in run.main.calls)
    assert sum(call["seconds"] for call in run.main.calls) < 25
    assert all(call["prompt"] is None for call in run.anchor.calls)

    samples = decode(tmp_path / "movie.clean.mkv")
    hell = next(d for d in report["detections"] if d["heard"] == "hell")
    assert tone_gain(samples, (hell["start"] + hell["end"]) / 2) < 0.01
    assert tone_gain(samples, 21.0) == pytest.approx(1.0, abs=0.1)

    # The subtitles it used get a censored copy next to the output, named so players load it.
    copy = tmp_path / "movie.clean.en.srt"
    assert result.subtitle_copy == copy
    assert copy.read_text("utf-8") == SUBTITLES.replace("the hell is", "the h*** is")
    assert report["output"]["subtitle_copy"] == {
        "source": str(tmp_path / "movie.en.srt"), "path": str(copy), "masked": 2
    }  # fmt: skip


def test_embedded_subtitles_are_used_first_and_censored_in_the_output(tmp_path: Path) -> None:
    source = clip(tmp_path, sidecar=None, embedded=SUBTITLES)
    run = Run(tmp_path, words=spoken(), **{"analysis.strategy": "targeted"})
    result, report = run.process(source, dry_run=False)
    assert report["subtitle"]["source"] == "embedded"
    assert report["strategy"]["used"] == "targeted"
    assert len(report["detections"]) == 2
    assert report["output"]["subtitles"] == [{"stream": 2, "codec": "subrip", "language": None, "masked": 2}]
    assert result.output is not None and result.subtitle_copy is None
    text = extract_subtitles(result.output)
    assert "What the h*** is going on" in text and "Get the f*** out" in text and "Freaking" in text


def test_explicit_subtitles_skip_the_search(tmp_path: Path) -> None:
    source = clip(tmp_path, sidecar=None)
    mine = tmp_path / "elsewhere" / "mine.srt"
    mine.parent.mkdir()
    mine.write_text(SUBTITLES, "utf-8")
    _, report = Run(tmp_path, words=spoken(), **{"analysis.strategy": "targeted"}).process(
        source, subtitles=mine
    )
    assert (report["subtitle"]["source"], report["subtitle"]["label"]) == ("explicit", "mine.srt")


def test_unconfirmed_masked_word_is_estimated_after_a_wider_look(tmp_path: Path) -> None:
    source = clip(tmp_path)
    heard = spoken(**{"Get the f*** out of my orchard now": "Get the frick out of my orchard now"})
    run = Run(tmp_path, words=heard, **{"analysis.strategy": "targeted"})
    _, report = run.process(source)
    assert report["confirmation"] == {"strong_flags": 2, "confirmed": 1}
    assert report["windows"]["expanded"] == 1
    assert report["unconfirmed"] == [
        {"cue": 4, "text": "Get the f*** out of my orchard now", "resolution": "estimate"}
    ]
    estimate = next(d for d in report["detections"] if d["source"] == "estimate")
    # "f***" is characters 8-12 of 34, spread over the cue (30.2-33.2 s after sync), widened by 0.3 s
    assert estimate["start"] == pytest.approx(30.2 + 3 * 8 / 34 - 0.3, abs=0.03)
    assert estimate["end"] == pytest.approx(30.2 + 3 * 12 / 34 + 0.3, abs=0.03)
    assert (estimate["heard"], estimate["cue"]) == ("f***", 4)
    assert len(run.main.calls) == 4  # three windows, then the wider re-check


def test_unconfirmed_words_can_be_skipped(tmp_path: Path) -> None:
    source = clip(tmp_path)
    heard = spoken(**{"Get the f*** out of my orchard now": "Get the frick out of my orchard now"})
    settings = {"analysis.strategy": "targeted", "analysis.targeted.on_unconfirmed": "skip"}
    _, report = Run(tmp_path, words=heard, **settings).process(source)
    assert [d["heard"] for d in report["detections"]] == ["hell"]
    assert report["unconfirmed"][0]["resolution"] == "skip"


def test_mostly_unconfirmed_flags_escalate_to_full(tmp_path: Path) -> None:
    lines = [
        (2.0, 5.0, "We found the copper lantern near the harbor"),
        (10.0, 13.0, "What the hell is going on here"),
        (20.0, 23.0, "Hell no I will not go"),
        (30.0, 33.0, "Oh hell the violin broke again"),
        (40.0, 43.0, "Meadow falcons never sleep before dawn"),
    ]
    source = clip(tmp_path, sidecar=srt(*lines))
    heard = spoken(
        lines, **{text: text.replace("hell", "heck").replace("Hell", "Heck") for _, _, text in lines}
    )
    run = Run(tmp_path, words=heard, **{"analysis.strategy": "targeted"})
    _, report = run.process(source)
    assert report["strategy"]["used"] == "full"
    assert report["strategy"]["fallback_reason"] == (
        "speech recognition confirmed only 0 of 3 flagged cues, so the subtitles do not match this audio"
    )
    assert report["confirmation"] == {"strong_flags": 3, "confirmed": 0}
    assert run.main.calls[-1]["seconds"] == pytest.approx(DURATION, abs=0.1)  # the whole track


def test_subtitles_out_of_sync_fall_back_to_full(tmp_path: Path) -> None:
    source = clip(tmp_path)
    late = [Word(w.text, w.start + 8.0, w.end + 8.0, w.probability) for w in spoken() if w.end + 8 < DURATION]
    run = Run(tmp_path, words=late, **{"analysis.strategy": "targeted"})
    _, report = run.process(source)
    assert report["strategy"]["used"] == "full"
    [candidate] = report["subtitle_candidates"]
    assert "anchor cues were heard" in candidate["result"]
    assert report["strategy"]["fallback_reason"].startswith("no usable subtitles: movie.en.srt: only")


def test_sync_failure_is_an_error_without_fallback(tmp_path: Path) -> None:
    source = clip(tmp_path)
    run = Run(tmp_path, words=[], **{"analysis.strategy": "targeted", "analysis.fallback_to_full": False})
    with pytest.raises(VbrError, match=r"no usable subtitles.*fallback_to_full is false"):
        run.process(source)


def test_coverage_guard(tmp_path: Path) -> None:
    source = clip(tmp_path)
    settings = {"analysis.strategy": "targeted", "analysis.targeted.max_coverage": 0.1}
    _, report = Run(tmp_path, words=spoken(), **settings).process(source)
    assert report["strategy"]["used"] == "full"
    # three windows, each a flagged cue plus 1.5 s on either side: 17 s of the 60 s
    assert report["strategy"]["fallback_reason"] == (
        "the windows would cover 28% of the runtime (more than max_coverage, 10%), "
        "so transcribing everything is cheaper"
    )
    assert report["windows"]["count"] == 3

    settings["analysis.fallback_to_full"] = False  # the windows are still cheaper than everything: run them
    _, report = Run(tmp_path, words=spoken(), **settings).process(source)
    assert report["strategy"]["used"] == "targeted"


def test_clean_subtitles_need_no_transcription_beyond_the_sync_check(tmp_path: Path) -> None:
    lines = [LINES[0], LINES[2], LINES[4], LINES[6]]
    source = clip(tmp_path, sidecar=srt(*lines))
    run = Run(tmp_path, words=spoken(lines), **{"analysis.strategy": "targeted"})
    result, report = run.process(source)
    assert (result.strategy, result.detections) == ("targeted", 0)
    assert run.main.calls == []  # the main model is never even loaded
    assert report["transcription"]["model"] is None
    assert report["windows"]["count"] == 0


def test_hybrid_also_transcribes_speech_no_cue_covers(tmp_path: Path) -> None:
    source = clip(tmp_path)
    song = say("you lying bastard", 56.0, 57.5)  # not in the subtitles
    run = Run(tmp_path, words=spoken(), extra=song, speech=[(55.8, 57.8), (10.0, 13.0)])
    _, report = run.process(source)
    assert report["strategy"] == {"requested": "hybrid", "used": "hybrid", "fallback_reason": None}
    assert report["windows"]["uncovered_regions"] == 1
    assert [d["heard"] for d in report["detections"]] == ["hell", "fuck", "bastard"]
    assert report["detections"][-1]["cue"] is None
    assert "decode" in report["timings"] and "vad" in report["timings"]


def test_a_second_run_is_served_from_the_transcript_cache(tmp_path: Path) -> None:
    source = clip(tmp_path)
    # A short clip: one more window would cross max_coverage and switch to full.
    settings = {"analysis.strategy": "targeted", "analysis.targeted.max_coverage": 0.9}
    first, report = Run(tmp_path, words=spoken(), **settings).process(source)
    assert report["transcription"]["from_cache"] == "none"

    again = Run(tmp_path, words=spoken(), **settings)
    second, cached = again.process(source)
    assert again.loaded == [] and again.anchor.calls == [] and again.main.calls == []  # no model at all
    assert cached["transcription"]["from_cache"] == "all" and cached["windows"]["cached"] == 3
    assert cached["subtitle"]["sync"] == report["subtitle"]["sync"]
    assert cached["detections"] == report["detections"] and second.intervals == first.intervals == 2

    # A word added to the list flags one more cue: only its window is transcribed.
    edited = Run(tmp_path, words=spoken(), **settings, **{"lexicon.categories.extra.terms": ["copper"]})
    _, report = edited.process(source)
    # cue 1 (2.0-5.0 s, heard 0.2 s later) padded by 1.5 s: 0.7-6.7 s
    assert [(round(c["start"], 1), round(c["seconds"], 1)) for c in edited.main.calls] == [(0.7, 6.0)]
    assert edited.anchor.calls == [] and report["transcription"]["from_cache"] == "some"
    assert [d["heard"] for d in report["detections"]] == ["copper", "hell", "fuck"]


def test_hybrid_reuses_the_speech_it_found_and_skips_decoding(tmp_path: Path) -> None:
    source = clip(tmp_path)
    song = say("you lying bastard", 56.0, 57.5)
    first = Run(tmp_path, words=spoken(), extra=song, speech=[(55.8, 57.8)])
    _, report = first.process(source)
    assert first.vad_runs == 1 and "decode" in report["timings"]

    again = Run(tmp_path, words=spoken(), extra=song, speech=[(55.8, 57.8)])
    _, cached = again.process(source)
    assert again.vad_runs == 0 and "decode" not in cached["timings"]
    assert again.main.calls == [] and cached["windows"]["uncovered_regions"] == 1
    assert [d["heard"] for d in cached["detections"]] == ["hell", "fuck", "bastard"]


def test_the_transcript_cache_can_be_turned_off(tmp_path: Path) -> None:
    source = clip(tmp_path)
    settings = {"analysis.strategy": "targeted", "cache.transcripts": False}
    Run(tmp_path, words=spoken(), **settings).process(source)
    again = Run(tmp_path, words=spoken(), **settings)
    _, report = again.process(source)
    assert len(again.main.calls) == 3 and report["transcription"]["from_cache"] == "none"


def test_a_window_that_grew_transcribes_only_what_the_cache_lacks(tmp_path: Path) -> None:
    source = clip(tmp_path)
    settings = {"analysis.strategy": "targeted", "analysis.targeted.max_coverage": 0.9}
    Run(tmp_path, words=spoken(), **settings).process(source)  # hell window: 8.7-14.7 s

    # "pilgrim" flags cue 3 (20-23 s), and a wide merge gap joins its window with the hell and fuck
    # windows: 8.7-34.7 s. The cache covers 8.7-14.4 s and 29.0-34.7 s reliably, so only the middle is
    # transcribed, reaching 2 s into the cached pieces on either side.
    grown = {**settings, "analysis.targeted.merge_gap_s": 5.0, "lexicon.categories.extra.terms": ["pilgrim"]}
    edited = Run(tmp_path, words=spoken(), **grown)
    _, report = edited.process(source)
    assert [(round(c["start"], 1), round(c["seconds"], 1)) for c in edited.main.calls] == [(12.4, 18.6)]
    assert report["windows"]["partly_cached"] == 1 and report["transcription"]["from_cache"] == "some"
    assert [d["heard"] for d in report["detections"]] == ["hell", "pilgrim", "fuck"]


def test_the_wider_re_check_is_cached_too(tmp_path: Path) -> None:
    source = clip(tmp_path)
    heard = spoken(**{"Get the f*** out of my orchard now": "Get the frick out of my orchard now"})
    settings = {"analysis.strategy": "targeted"}
    Run(tmp_path, words=heard, **settings).process(source)
    again = Run(tmp_path, words=heard, **settings)
    _, report = again.process(source)
    assert again.main.calls == [] and report["windows"]["expanded"] == 1
    assert report["transcription"]["from_cache"] == "all"
    assert [d["source"] for d in report["detections"]] == ["asr", "estimate"]
