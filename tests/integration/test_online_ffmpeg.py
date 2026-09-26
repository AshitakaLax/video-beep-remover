"""OpenSubtitles end to end: real FFmpeg, mocked HTTP (respx), scripted speech recognition.

These are the M3 fixtures: subtitles that are out of sync or timed for another frame rate are
corrected, and a used-up download quota falls back cleanly."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from helpers import FakeTranscriber, StrictUI, make_clip, say, srt
from video_beep_remover.config import load_config
from video_beep_remover.pipeline import FileResult, Pipeline, RunOptions
from video_beep_remover.subtitles.opensubtitles import API
from video_beep_remover.subtitles.oshash import opensubtitles_hash

pytestmark = pytest.mark.ffmpeg

DURATION = 60.0
LINES = [
    (2.0, 5.0, "We found the copper lantern near the harbor"),
    (10.0, 13.0, "What the hell is going on over there"),
    (20.0, 23.0, "Every pilgrim needs a marble saddle today"),
    (30.0, 33.0, "Get the f*** out of my orchard now"),
    (40.0, 43.0, "The glacier violin sounds like thunder again"),
    (50.0, 53.0, "Meadow falcons never sleep before dawn here"),
]
SPOKEN = [w for start, end, text in LINES for w in say(text.replace("f***", "fuck"), start + 0.2, end - 0.3)]


def timed(transform: Callable[[float], float]) -> str:
    """The subtitles with every time passed through `transform` (true time -> subtitle time)."""
    return srt(*((transform(start), transform(end), text) for start, end, text in LINES))


def result(file_id: int, **attributes: Any) -> dict[str, Any]:
    return {
        "id": str(file_id),
        "attributes": {
            "language": "en",
            "files": [{"file_id": file_id, "file_name": f"{file_id}.srt"}],
            **attributes,
        },
    }


class Service:
    """A fake OpenSubtitles: hash search, title search and downloads, with a quota."""

    def __init__(
        self,
        router: respx.MockRouter,
        *,
        by_hash: list[Any],
        by_title: list[Any],
        files: dict[int, str],
        quota: int = 5,
    ):
        self.searches: list[dict[str, str]] = []
        self.downloads: list[int] = []
        self.quota = quota
        router.get(f"{API}/subtitles").mock(side_effect=self._search)
        router.post(f"{API}/download").mock(side_effect=self._download)
        for file_id, text in files.items():
            router.get(f"https://dl.test/{file_id}.srt").mock(
                return_value=httpx.Response(200, content=text.encode())
            )
        self.by_hash, self.by_title = by_hash, by_title

    def _search(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.searches.append(params)
        return httpx.Response(200, json={"data": self.by_hash if "moviehash" in params else self.by_title})

    def _download(self, request: httpx.Request) -> httpx.Response:
        file_id = json.loads(request.content)["file_id"]
        if len(self.downloads) >= self.quota:
            return httpx.Response(406, json={"remaining": 0, "message": "You have downloaded your allowed 5 subtitles for 24h",
                                             "reset_time_utc": "2026-09-27T01:02:03.000Z"})  # fmt: skip
        self.downloads.append(file_id)
        return httpx.Response(200, json={"link": f"https://dl.test/{file_id}.srt", "file_name": f"{file_id}.srt",
                                         "remaining": self.quota - len(self.downloads), "reset_time_utc": None})  # fmt: skip


def run(tmp_path: Path, source: Path, **overrides: Any) -> tuple[FileResult, dict[str, Any], FakeTranscriber]:
    settings = {
        "transcription.device": "cpu",
        "analysis.strategy": "targeted",
        "subtitles.opensubtitles.api_key": "test-key",
        **overrides,
    }
    loaded = load_config(None, env={}, cwd=tmp_path, overrides=settings)
    fake = FakeTranscriber(SPOKEN)

    def no_speech(
        audio: np.ndarray, on_progress: Callable[[float], None] | None = None
    ) -> list[tuple[float, float]]:
        return []

    pipeline = Pipeline(
        loaded, ui=StrictUI(), transcriber_factory=lambda choice: fake, speech_detector=no_speech
    )
    outcome = pipeline.process(source, RunOptions(dry_run=True, report=tmp_path / "report.json"))
    return outcome, json.loads((tmp_path / "report.json").read_text("utf-8")), fake


@pytest.fixture
def movie(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "The.Movie.2019.1080p.BluRay.x264-GRP.mkv", duration=DURATION)


def heard(report: dict[str, Any]) -> list[tuple[str, float]]:
    return [(d["heard"], round(d["start"], 1)) for d in report["detections"]]


@respx.mock
def test_hash_matched_subtitles_are_trusted_downloaded_once_and_cached(tmp_path: Path, movie: Path) -> None:
    service = Service(respx.mock, by_hash=[result(11, moviehash_match=True, download_count=50)], by_title=[],
                      files={11: timed(lambda t: t)})  # fmt: skip
    outcome, report, _ = run(tmp_path, movie)
    assert outcome.strategy == "targeted"
    assert report["subtitle"]["source"] == "opensubtitles" and report["subtitle"]["trusted"]
    assert heard(report) == [("hell", 10.8), ("fuck", 30.8)]
    assert service.searches == [
        {"languages": "en", "moviehash": opensubtitles_hash(movie)}
    ]  # no title search
    assert report["opensubtitles"]["downloads"] == 1
    assert report["input"]["oshash"] == opensubtitles_hash(movie)

    _, again, _ = run(tmp_path, movie)
    assert service.downloads == [11]  # the second run used the cache
    assert again["subtitle_candidates"][0]["cached"] is True

    respx.mock.reset()
    _, offline, _ = run(tmp_path, movie, offline=True)  # no network at all: the cache still serves it
    assert offline["subtitle"]["file_id"] == 11
    assert not respx.mock.calls


@respx.mock
def test_out_of_sync_subtitles_from_a_title_search_are_corrected(tmp_path: Path, movie: Path) -> None:
    """Not timed for this file (no hash match), and 12 s early: the ±20 s search finds the offset."""
    service = Service(respx.mock, by_hash=[], by_title=[result(21, release="The.Movie.2019.720p.WEB-OTHER")],
                      files={21: timed(lambda t: t + 12.0)})  # fmt: skip
    outcome, report, _ = run(tmp_path, movie)
    assert outcome.strategy == "targeted"
    assert [s.get("query") for s in service.searches] == [None, "the movie"]
    assert service.searches[1]["year"] == "2019"
    sync = report["subtitle"]["sync"]
    assert not report["subtitle"]["trusted"]
    assert sync["offset"] == pytest.approx(-12.0 + 0.2, abs=0.05) and sync["scale"] == 1.0
    assert heard(report) == [("hell", 10.8), ("fuck", 30.8)]


@respx.mock
def test_subtitles_for_another_frame_rate_are_rescaled_up_front(tmp_path: Path, movie: Path) -> None:
    """Timed for a 25 fps release of this 24 fps video: every time is 4 % early, more at the end."""
    service = Service(
        respx.mock, by_hash=[], by_title=[result(31, fps=25.0)], files={31: timed(lambda t: t * 24 / 25)}
    )
    outcome, report, _ = run(tmp_path, movie)
    assert outcome.strategy == "targeted" and service.downloads == [31]
    [tried] = report["subtitle_candidates"]
    assert tried["frame_rate_ratio"] == 25 / 24
    assert report["subtitle"]["sync"]["scale"] == 25 / 24
    assert heard(report) == [("hell", 10.8), ("fuck", 30.8)]


@respx.mock
def test_a_used_up_quota_falls_back_cleanly(tmp_path: Path, movie: Path) -> None:
    Service(respx.mock, by_hash=[result(41, moviehash_match=True), result(42)], by_title=[],
            files={41: timed(lambda t: t)}, quota=0)  # fmt: skip
    outcome, report, fake = run(tmp_path, movie)
    assert outcome.strategy == "full"
    assert report["strategy"]["fallback_reason"].startswith("no usable subtitles: OpenSubtitles #41")
    assert "download quota is used up" in report["subtitle_candidates"][0]["result"]
    assert report["subtitle_candidates"][1]["result"].startswith(
        "unusable: not downloaded"
    )  # no second request
    assert report["opensubtitles"]["quota"] == {"remaining": 0, "reset_time_utc": "2026-09-27T01:02:03.000Z"}
    assert fake.calls[-1]["seconds"] == pytest.approx(DURATION, abs=0.1)
    assert heard(report) == [("hell", 10.8), ("fuck", 30.8)]  # full transcription found them anyway


@respx.mock
def test_without_a_key_only_cached_subtitles_are_used(tmp_path: Path, movie: Path) -> None:
    outcome, report, _ = run(tmp_path, movie, **{"subtitles.opensubtitles.api_key": ""})
    assert outcome.strategy == "full"
    assert report["subtitle_search"] == [
        "OpenSubtitles: skipped, no API key (set OPENSUBTITLES_API_KEY to your own free key)"
    ]
    assert not respx.mock.calls


@respx.mock
def test_ffsubsync_rescues_subtitles_too_far_out_of_sync(
    tmp_path: Path, movie: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """40 s late is beyond the ±20 s search; ffsubsync (a stand-in here) re-times them."""
    Service(respx.mock, by_hash=[], by_title=[result(51)], files={51: timed(lambda t: t + 40.0)})
    stub = tmp_path / "fake_ffsubsync.py"  # stands in for ffsubsync: moves every cue 40 s earlier
    stub.write_text(
        "import sys, pysubs2\n"
        "args = sys.argv[1:]\n"
        "subs = pysubs2.load(args[args.index('-i') + 1])\n"
        "subs.shift(s=-40)\n"
        "subs.save(args[args.index('-o') + 1])\n",
        "utf-8",
    )
    monkeypatch.setattr(
        "video_beep_remover.subtitles.ffsubsync.ffsubsync_command", lambda: [sys.executable, str(stub)]
    )
    outcome, report, _ = run(tmp_path, movie)
    assert outcome.strategy == "targeted"
    [tried] = report["subtitle_candidates"]
    assert tried["ffsubsync"] == "in sync"
    assert "anchor cues were heard" in tried["first_sync"]["reason"]
    assert heard(report) == [("hell", 10.8), ("fuck", 30.8)]

    monkeypatch.setattr("video_beep_remover.subtitles.ffsubsync.ffsubsync_command", lambda: None)
    _, without, _ = run(tmp_path, movie)  # without ffsubsync the same file falls back to full
    assert without["strategy"]["used"] == "full"
