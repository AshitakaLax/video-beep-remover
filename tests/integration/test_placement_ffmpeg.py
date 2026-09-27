"""Where the cleaned file goes (outputs.py): --backup and --in-place, through the pipeline and the command
line, with real FFmpeg and a scripted transcriber."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import FakeTranscriber, StrictUI, decode, make_clip, tone_gain, words
from video_beep_remover.cli import app
from video_beep_remover.config import load_config
from video_beep_remover.errors import RenderError, UsageError
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe
from video_beep_remover.pipeline import Pipeline, RunOptions

pytestmark = pytest.mark.ffmpeg
runner = CliRunner()
SPOKEN = words(("well", 1.0, 1.3), ("damn", 2.0, 2.4))


def pipeline(tmp_path: Path, mode: str) -> Pipeline:
    overrides: dict[str, Any] = {"transcription.device": "cpu", "output.mode": mode}
    loaded = load_config(None, env={}, cwd=tmp_path, overrides=overrides)
    return Pipeline(loaded, ui=StrictUI(), transcriber_factory=lambda choice: FakeTranscriber(SPOKEN))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names(folder: Path, suffix: str) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.suffix == suffix)


@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
def test_backup_keeps_the_original_and_cleans_in_its_place(tmp_path: Path, suffix: str) -> None:
    source = make_clip(tmp_path / f"movie{suffix}")
    original = digest(source)
    result = pipeline(tmp_path, "backup").process(source, RunOptions(edl=True))
    backup = tmp_path / f"movie.orig{suffix}"
    assert (result.status, result.output, result.backup) == ("cleaned", source, backup)
    assert digest(backup) == original  # unmodified
    assert tone_gain(decode(source), 2.2) < 0.01
    assert tone_gain(decode(source), 1.1) == pytest.approx(1.0, abs=0.1)
    assert "vbr_censored" in probe(FFmpeg(), source).tags  # so folder runs skip it, in MP4 too
    report = json.loads((tmp_path / "movie.vbr.json").read_text("utf-8"))
    assert report["output"]["path"] == str(source) and report["output"]["backup"] == str(backup)
    assert (tmp_path / "movie.orig.edl").is_file()  # the EDL mutes the original, where it now is
    assert names(tmp_path, suffix) == sorted([f"movie{suffix}", f"movie.orig{suffix}"])  # no .partial

    again = pipeline(tmp_path, "backup").process(source, RunOptions())
    assert again.status == "skipped" and "movie.orig" in again.notes[0]
    assert digest(backup) == original  # a backup is never overwritten


def test_in_place_replaces_the_original(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv")
    result = pipeline(tmp_path, "in_place").process(source, RunOptions())
    assert (result.status, result.output, result.backup) == ("cleaned", source, None)
    assert tone_gain(decode(source), 2.2) < 0.01
    assert names(tmp_path, ".mkv") == ["movie.mkv"]


def test_a_failed_render_leaves_the_original_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_clip(tmp_path / "movie.mp4")
    original = digest(source)
    monkeypatch.setattr(
        "video_beep_remover.media.render.verify_muted", lambda *args, **kwargs: (1, ["1.88-2.60 s: loud"])
    )
    with pytest.raises(RenderError, match="verification failed"):
        pipeline(tmp_path, "backup").process(source, RunOptions())
    assert digest(source) == original
    assert names(tmp_path, ".mp4") == ["movie.mp4"]


def test_backup_and_in_place_do_not_take_an_output_path(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv", duration=1.0)
    with pytest.raises(UsageError, match="-o cannot be combined with --backup"):
        pipeline(tmp_path, "backup").process(source, RunOptions(output=tmp_path / "out.mkv"))


def test_scan_then_render_and_clean_a_folder_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Pipeline, "_load_transcriber", lambda self, choice: FakeTranscriber(SPOKEN))
    monkeypatch.setattr("video_beep_remover.pipeline.silero_speech", lambda audio, on_progress=None: [])
    for name in ("a.mp4", "b.mkv"):
        make_clip(tmp_path / name, duration=3.0)
    scanned = runner.invoke(app, ["scan", str(tmp_path), "--device", "cpu"])
    assert scanned.exit_code == 0, scanned.output
    make_clip(tmp_path / "c.mkv", duration=3.0)  # added after the scan: it has no report

    rendered = runner.invoke(app, ["render", str(tmp_path), "--backup"])
    assert rendered.exit_code == 0, rendered.output
    assert "c.mkv: skipped (no report" in rendered.output
    assert "a.mp4: muted 1 spans in place (from the report)" in rendered.output
    assert tone_gain(decode(tmp_path / "a.mp4"), 2.2) < 0.01
    assert tone_gain(decode(tmp_path / "a.orig.mp4"), 2.2) == pytest.approx(1.0, abs=0.1)

    cleaned = runner.invoke(app, ["clean", str(tmp_path), "--backup", "--device", "cpu"])
    assert cleaned.exit_code == 0, cleaned.output
    assert "a.orig.mp4: skipped (it is the backup of a.mp4)" in cleaned.output
    assert "a.mp4: skipped" in cleaned.output and "b.mkv: skipped" in cleaned.output
    assert "c.mkv: muted 1 spans in place" in cleaned.output
    assert names(tmp_path, ".mkv") == ["b.mkv", "b.orig.mkv", "c.mkv", "c.orig.mkv"]

    both = runner.invoke(app, ["clean", str(tmp_path), "--backup", "--in-place"])
    assert both.exit_code == 2 and "cannot be combined" in both.output
