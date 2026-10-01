"""`vbr scan --review-srt` and `vbr render --report` through the command line, with real FFmpeg."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import FakeTranscriber, decode, make_clip, tone_gain, words
from video_beep_remover.cli import app
from video_beep_remover.model_pool import ModelPool

pytestmark = pytest.mark.ffmpeg
runner = CliRunner()


@pytest.fixture(autouse=True)
def scripted_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    heard = words(("well", 1.0, 1.3), ("damn", 2.0, 2.4))
    monkeypatch.setattr(ModelPool, "_load_transcriber", lambda self, choice: FakeTranscriber(heard))
    monkeypatch.setattr("video_beep_remover.pipeline.silero_speech", lambda audio, on_progress=None: [])


def test_scan_then_render_an_edited_report(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    scanned = runner.invoke(app, ["scan", str(source), "--review-srt", "--device", "cpu"])
    assert scanned.exit_code == 0, scanned.output
    assert "review subtitles:" in scanned.output
    assert (tmp_path / "movie.review.srt").read_text("utf-8").endswith("[muted] damn\n")

    report_path = tmp_path / "movie.vbr.json"
    report = json.loads(report_path.read_text("utf-8"))
    report["intervals"].append({"start": 4.0, "end": 4.5})  # also mute something the scan did not find
    report_path.write_text(json.dumps(report), "utf-8")

    rendered = runner.invoke(app, ["render", str(source), "--report", str(report_path), "--review-srt"])
    assert rendered.exit_code == 0, rendered.output
    assert "muted 2 spans" in rendered.output and "from the report" in rendered.output
    samples = decode(tmp_path / "movie.clean.mkv")
    assert tone_gain(samples, 2.2) < 0.01 and tone_gain(samples, 4.25) < 0.01
    assert tone_gain(samples, 3.2, window=0.05) == pytest.approx(1.0, abs=0.05)  # untouched between them
    review = (tmp_path / "movie.clean.review.srt").read_text("utf-8")
    assert review.count("[muted]") == 2 and "[muted] damn" in review

    again = runner.invoke(app, ["render", str(source), "--report", str(report_path)])
    assert again.exit_code == 2 and "already exists" in again.output


def test_render_refuses_a_report_for_another_file(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv", duration=4.0)
    report = tmp_path / "other.vbr.json"
    report.write_text(
        json.dumps({"input": {"duration": 60.0}, "intervals": [{"start": 1, "end": 2}]}), "utf-8"
    )
    refused = runner.invoke(app, ["render", str(source), "--report", str(report)])
    assert refused.exit_code == 2
    assert "made for a different file" in refused.output and "--force" in refused.output
    forced = runner.invoke(
        app, ["render", str(source), "--report", str(report), "--force", "-o", str(tmp_path / "o.mkv")]
    )
    assert forced.exit_code == 0, forced.output
    assert tone_gain(decode(tmp_path / "o.mkv"), 1.5) < 0.01
