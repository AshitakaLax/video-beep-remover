"""Context analysis in the pipeline (DESIGN.md §17), with real FFmpeg and stand-in models: the verdicts
reach the report and the review subtitles, and what is muted does not change."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from helpers import FakeTranscriber, StrictUI, make_clip, say, srt
from video_beep_remover.config import load_config
from video_beep_remover.context.models import LABELS, Question
from video_beep_remover.models import Word
from video_beep_remover.pipeline import Pipeline, RunOptions

pytestmark = pytest.mark.ffmpeg

DURATION = 40.0
LEAD = 0.2
LINES = [
    (2.0, 5.0, "We found the copper lantern near the harbor"),
    (8.0, 11.0, "Go to hell and take your lantern with you"),
    (14.0, 17.0, "The road to hell is paved with good intentions"),
    (20.0, 23.0, "Every pilgrim needs a marble saddle today"),
    (26.0, 29.0, "Did you sleep with the captain last winter"),
    (32.0, 35.0, "The glacier violin sounds like thunder again"),
]
SUBTITLES = srt(*LINES[:4], (24.0, 25.0, "[moaning]"), *LINES[4:])


class Classifier:
    name = "fake-classifier"

    def score(self, texts: Sequence[str]) -> list[dict[str, float]]:
        # Ambiguous words reach the classifier masked; the rest of this line is what makes it rude.
        return [
            dict.fromkeys(LABELS, 0.0) | ({"toxicity": 0.95} if "Go to [...]" in t else {}) for t in texts
        ]


class Judge:
    name = "fake-judge"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: Question) -> str:
        prompt = question.text
        self.questions.append(prompt)
        line = next(part for part in prompt.splitlines() if part.startswith(">> "))
        if "road to hell" in line:
            return '{"use": "harmless", "reason": "place", "emotion": "neutral"}'
        return '{"sexual": true}' if "sleep with" in line else '{"sexual": false}'


def heard() -> list[Word]:
    return [word for start, end, text in LINES for word in say(text, start + LEAD, end - 0.3)]


def scan(tmp_path: Path, **overrides: Any) -> tuple[Pipeline, dict[str, Any], str]:
    loaded = load_config(
        None,
        env={},
        cwd=tmp_path,
        overrides={"transcription.device": "cpu", "analysis.strategy": "targeted", **overrides},
    )
    words = heard()
    judge = Judge()
    pipeline = Pipeline(
        loaded,
        ui=StrictUI(),
        transcriber_factory=lambda choice: FakeTranscriber(words),
        context_models=(lambda name, device: Classifier(), lambda name, device: judge),
    )
    name = "on" if loaded.config.context.enabled else "off"
    result = pipeline.process(
        tmp_path / "movie.mkv", RunOptions(dry_run=True, review_srt=True, report=tmp_path / f"{name}.json")
    )
    assert result.review is not None
    review = result.review.read_text("utf-8")
    result.review.unlink()
    return pipeline, json.loads((tmp_path / f"{name}.json").read_text("utf-8")), review


def test_context_verdicts_are_reported_and_change_nothing_muted(tmp_path: Path) -> None:
    make_clip(tmp_path / "movie.mkv", duration=DURATION)
    (tmp_path / "movie.srt").unlink(missing_ok=True)
    (tmp_path / "movie.en.srt").write_text(SUBTITLES, "utf-8")
    _, plain, _ = scan(tmp_path)
    overrides = {"context.enabled": True, "context.judge": "fake-judge"}
    pipeline, report, review = scan(tmp_path, **overrides)
    assert "context" not in plain and all("context" not in d for d in plain["detections"])
    assert report["intervals"] == plain["intervals"]  # report-only

    verdicts = [(d["heard"], d["context"]["use"], d["context"]["reason"]) for d in report["detections"]]
    assert verdicts == [("hell", "profane", "classifier"), ("hell", "harmless", "place")]
    harmless = report["detections"][1]["context"]
    assert (harmless["action"], harmless["line"], harmless["judged"]) == (
        "keep",
        "The road to hell is paved with good intentions",
        True,
    )
    context = report["context"]
    assert (context["classifier"], context["judge"], context["judge_questions"]) == (
        "unitary/unbiased-toxic-roberta",
        "fake-judge",
        2,  # the harmless-looking "hell", and the ambiguous "sleep with"
    )
    flagged = [(s["text"], s["sounds"], s["certain"], s["evidence"]) for s in context["sexual_lines"]]
    assert flagged == [
        ("", ["moaning"], False, ["sound [moaning]"]),
        (
            "Did you sleep with the captain last winter",
            [],
            True,
            ['phrase "sleep with" (ambiguous)', "judge"],
        ),
    ]
    assert "[muted] hell (probably harmless: place)" in review
    assert "[possibly sexual] sound [moaning]" in review
    assert '[sexual line] phrase "sleep with" (ambiguous); judge' in review
    assert "context" in report["timings"]

    # The answers are cached: a second run asks nothing.
    again = pipeline.process(tmp_path / "movie.mkv", RunOptions(dry_run=True, report=tmp_path / "again.json"))
    assert again.report is not None
    assert json.loads(again.report.read_text("utf-8"))["context"]["judge_questions"] == 0


def test_context_actions_keep_harmless_uses_and_mute_sexual_lines(tmp_path: Path) -> None:
    make_clip(tmp_path / "movie.mkv", duration=DURATION)
    (tmp_path / "movie.srt").unlink(missing_ok=True)
    (tmp_path / "movie.en.srt").write_text(SUBTITLES, "utf-8")
    _, plain, _ = scan(tmp_path)
    overrides = {
        "context.enabled": True,
        "context.judge": "fake-judge",
        "context.harmless": "keep",
        "context.sexual": "mute",
    }
    pipeline, report, review = scan(tmp_path, **overrides)

    context = report["context"]
    assert (context["harmless"], context["sexual"], context["kept"]) == ("keep", "mute", 1)
    moaning, line = context["sexual_lines"]
    assert "muted" not in moaning  # only "possibly" sexual: a sound with nothing said
    # No window covered the sexual line, which holds no listed word: it was transcribed on its own.
    first, *_, last = say(LINES[4][2], LINES[4][0] + LEAD, LINES[4][1] - 0.3)
    assert line["muted"] == {"start": round(first.start, 3), "end": round(last.end, 3), "from": "heard"}

    # The harmless "hell" is no longer muted; the rude one is, and so is the sexual line, padded like a word.
    rude = plain["intervals"][0]
    assert report["intervals"][0] == rude and len(report["intervals"]) == 2
    muted = report["intervals"][1]
    assert muted["start"] == pytest.approx(first.start - 0.12, abs=1e-3)
    assert muted["end"] == pytest.approx(last.end + 0.2, abs=1e-3)

    assert "[kept] hell (probably harmless: place)" in review
    assert '[muted] sexual line (phrase "sleep with" (ambiguous); judge)' in review
    assert "[sexual line]" not in review  # the muted line needs no cue of its own
    assert "[possibly sexual] sound [moaning]" in review

    # vbr render reads the verdicts back from the report, and writes the same review cues.
    rendered = pipeline.render_report(
        tmp_path / "movie.mkv", tmp_path / "on.json", RunOptions(review_srt=True)
    )
    assert rendered.review is not None
    assert cue_texts(rendered.review.read_text("utf-8")) == cue_texts(review)


def cue_texts(subtitles: str) -> list[str]:
    return [block.splitlines()[2] for block in subtitles.strip().split("\n\n")]
