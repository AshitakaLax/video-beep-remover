"""The whole pipeline with real FFmpeg and a scripted transcriber."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers import FakeTranscriber, StrictUI, Track, decode, make_clip, tone_gain, words
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
    assert report["intervals"] == [{"start": 1.88, "end": 2.6}]  # 120 ms before the word, 200 ms after
    assert report["output"]["verified_spans"] == 1
    assert (tmp_path / "movie.edl").read_text("utf-8") == "1.880\t2.600\t1\n"


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


def test_review_subtitles_name_each_muted_span(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    run = pipeline(tmp_path)
    scanned = run.process(source, RunOptions(dry_run=True, review_srt=True))
    assert scanned.review == tmp_path / "movie.review.srt"
    assert scanned.review.read_text("utf-8") == "1\n00:00:01,880 --> 00:00:02,600\n[muted] damn\n"

    cleaned = run.process(source, RunOptions(review_srt=True))
    assert cleaned.review == tmp_path / "movie.clean.review.srt"
    shift = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))["output"]["timeline_shift"]
    start, end = (f"00:00:0{t + shift:.3f}".replace(".", ",") for t in (1.88, 2.6))
    assert cleaned.review.read_text("utf-8") == f"1\n{start} --> {end}\n[muted] damn\n"


def test_render_mutes_the_spans_of_a_hand_edited_report(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    run = pipeline(tmp_path)
    run.process(source, RunOptions(dry_run=True))
    report_path = tmp_path / "movie.vbr.json"
    report = json.loads(report_path.read_text("utf-8"))
    report["intervals"] = [{"start": 4.5, "end": 5.0}, {"start": 4.0, "end": 4.6}]  # "damn" is let through
    report_path.write_text(json.dumps(report), "utf-8")

    result = run.render_report(source, report_path, RunOptions())
    assert (result.status, result.output, result.intervals) == ("cleaned", tmp_path / "movie.clean.mkv", 1)
    samples = decode(tmp_path / "movie.clean.mkv")
    assert tone_gain(samples, 4.8) < 0.01  # the two spans overlap, so they merge: 4.0-5.0
    assert tone_gain(samples, 2.2, window=0.05) == pytest.approx(1.0, abs=0.05)
    assert json.loads(report_path.read_text("utf-8")) == report  # the report is only read

    other = make_clip(tmp_path / "other.mkv", duration=9.0)
    with pytest.raises(UsageError, match="made for a different file"):
        run.render_report(other, report_path, RunOptions())
    assert run.render_report(other, report_path, RunOptions(), force=True).status == "cleaned"


@pytest.mark.parametrize(
    ("intervals", "message"),
    [(None, 'no "intervals" list'), ([{"start": 2}], 'intervals\\[0\\]: needs a numeric "start" and "end"'),
     ([{"start": 3, "end": 2}], "end must be after start")],
)  # fmt: skip
def test_render_rejects_unusable_intervals(tmp_path: Path, intervals: Any, message: str) -> None:
    source = make_clip(tmp_path / "movie.mkv", duration=2.0)
    report = tmp_path / "edited.json"
    report.write_text(json.dumps({"intervals": intervals} if intervals is not None else {}), "utf-8")
    with pytest.raises(UsageError, match=message):
        pipeline(tmp_path).render_report(source, report, RunOptions())


def test_same_language_audio_is_muted_only_if_it_carries_the_same_dialogue(tmp_path: Path) -> None:
    tracks = [
        Track(noise_seed=1, default=True),  # analysed
        Track(noise_seed=1, title="Stereo"),  # the same audio: a downmix
        Track(noise_seed=2, title="Mislabelled dub"),  # other dialogue, also tagged English
        Track(noise_seed=1, language="fra"),  # another language: dropped whatever it carries
    ]
    source = make_clip(tmp_path / "movie.mkv", tracks=tracks)
    result = pipeline(tmp_path).process(source, RunOptions())
    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    checks = {c["stream"]: c for c in report["output"]["audio_checks"]}
    assert checks[2]["same_dialogue"] and checks[2]["correlation"] > 0.9
    assert not checks[3]["same_dialogue"] and checks[3]["correlation"] < 0.2
    assert 4 not in checks
    assert result.output is not None
    kept = probe(FFmpeg(), result.output).audio_streams
    assert [s.title for s in kept] == [None, "Stereo"]
    assert any(
        "dropped audio stream #3" in note and "does not carry the dialogue" in note for note in result.notes
    )
    assert not any("#3" in note and "gets the same mutes" in note for note in result.notes)
    for stream in ("0:a:0", "0:a:1"):
        samples = decode(result.output, stream)
        assert float(np.sqrt(np.mean(samples[int(2.0 * 48_000) : int(2.4 * 48_000)] ** 2))) < 1e-3


def test_a_full_transcript_is_cached_and_serves_the_next_run(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv", duration=12.0)  # big enough for a fingerprint
    fake = FakeTranscriber(SPOKEN)
    loaded = load_config(None, env={}, cwd=tmp_path, overrides={"analysis.strategy": "full"})
    Pipeline(loaded, transcriber_factory=lambda choice: fake).process(source, RunOptions(dry_run=True))
    assert len(fake.calls) == 1

    Pipeline(loaded, transcriber_factory=lambda choice: fake).process(source, RunOptions(dry_run=True))
    report = json.loads((tmp_path / "movie.vbr.json").read_text("utf-8"))
    assert len(fake.calls) == 1 and "decode" not in report["timings"]
    assert report["transcription"]["from_cache"] == "all"
    assert [d["heard"] for d in report["detections"]] == ["damn"]
