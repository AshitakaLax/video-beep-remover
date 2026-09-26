"""The whole pipeline with real FFmpeg and a scripted transcriber."""

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import FakeTranscriber, StrictUI, decode, make_clip, tone_gain, words
from video_beep_remover.config import load_config
from video_beep_remover.errors import UsageError, VbrError
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe
from video_beep_remover.pipeline import Pipeline, RunOptions

pytestmark = pytest.mark.ffmpeg

SPOKEN = words(("this", 1.0, 1.2), ("is", 1.25, 1.4), ("damn", 2.0, 2.4), ("good", 2.5, 2.8))


def pipeline(tmp_path: Path, spoken: list[Any] = SPOKEN, config: str = "", **overrides: Any) -> Pipeline:
    path = tmp_path / "vbr.toml"
    path.write_text(config, "utf-8")
    loaded = load_config(path, env={}, overrides={"transcription.device": "cpu", **overrides})
    fake = FakeTranscriber(spoken)
    return Pipeline(loaded, ui=StrictUI(), transcriber_factory=lambda choice: fake)


def test_clean_mutes_detections_and_writes_report_and_edl(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mp4")
    result = pipeline(tmp_path).process(source, RunOptions(edl=True))
    assert result.status == "cleaned"
    assert result.output == tmp_path / "movie.clean.mp4"
    samples = decode(result.output)
    assert tone_gain(samples, 2.2) < 0.01  # "damn" 2.0-2.4 plus padding
    assert tone_gain(samples, 1.1) == pytest.approx(1.0, abs=0.1)

    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    assert report["strategy"] == {
        "requested": "hybrid",
        "used": "full",
        "fallback_reason": (
            "no subtitles found (OpenSubtitles: skipped, no API key (set OPENSUBTITLES_API_KEY to your own free key))"
        ),
    }
    assert report["subtitle_candidates"] == []
    [detection] = report["detections"]
    assert (detection["heard"], detection["category"]) == ("damn", "mild")
    assert report["intervals"] == [{"start": 1.88, "end": 2.52}]
    assert report["output"]["verified_spans"] == 1
    assert (tmp_path / "movie.edl").read_text("utf-8") == "1.880\t2.520\t1\n"


def test_scan_writes_only_a_report(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    result = pipeline(tmp_path).process(source, RunOptions(dry_run=True))
    assert result.status == "scanned" and result.output is None
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix in (".mkv", ".json")) == [
        "movie.mkv",
        "movie.vbr.json",
    ]


def test_clean_file_is_copied_without_re_encoding(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    result = pipeline(tmp_path, spoken=words(("hello", 1.0, 1.3))).process(source, RunOptions())
    assert result.status == "copied"
    assert result.output is not None
    before, after = probe(FFmpeg(), source), probe(FFmpeg(), result.output)
    assert after.audio_streams[0].codec == before.audio_streams[0].codec == "aac"


def test_clean_file_can_be_skipped(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    run = pipeline(tmp_path, spoken=[], config='[output]\nwhen_clean = "skip"\n')
    result = run.process(source, RunOptions())
    assert result.status == "clean"
    assert not (tmp_path / "movie.clean.mkv").exists()


def test_existing_output_needs_overwrite_or_skip(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    (tmp_path / "movie.clean.mkv").write_bytes(b"old")
    run = pipeline(tmp_path)
    with pytest.raises(UsageError, match="already exists"):
        run.process(source, RunOptions())
    assert run.process(source, RunOptions(skip_existing=True)).status == "skipped"
    assert run.process(source, RunOptions(overwrite=True)).status == "cleaned"
    assert (tmp_path / "movie.clean.mkv").stat().st_size > 3


def test_no_fallback_fails_without_subtitles(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    run = pipeline(tmp_path, **{"analysis.fallback_to_full": False})
    with pytest.raises(VbrError, match=r"no subtitles found .*; fallback_to_full is false"):
        run.process(source, RunOptions())


def test_existing_edl_is_not_overwritten(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    (tmp_path / "movie.edl").write_text("10 20 3\n", "utf-8")  # e.g. a DVR's commercial markers
    result = pipeline(tmp_path).process(source, RunOptions(dry_run=True, edl=True))
    assert result.edl is None
    assert (tmp_path / "movie.edl").read_text("utf-8") == "10 20 3\n"


def test_prompt_and_language_reach_the_transcriber(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv", duration=3.0)
    fake = FakeTranscriber([])
    loaded = load_config(None, env={}, cwd=tmp_path, overrides={"transcription.device": "cpu"})
    Pipeline(loaded, transcriber_factory=lambda choice: fake).process(source, RunOptions(dry_run=True))
    [call] = fake.calls
    assert call["language"] == "en"
    assert call["seconds"] == pytest.approx(3.0, abs=0.05)
    assert isinstance(call["prompt"], str) and call["prompt"].startswith("Fuck")
