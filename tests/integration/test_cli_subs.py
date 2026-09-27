"""`vbr subs` and `--subtitles` through the command line, with real FFmpeg and scripted speech."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import FakeTranscriber, make_clip, say, srt
from video_beep_remover.cli import app
from video_beep_remover.models import Word
from video_beep_remover.pipeline import Pipeline

pytestmark = pytest.mark.ffmpeg
runner = CliRunner()

LINES = [
    (1.0, 3.5, "We found the copper lantern near the harbor"),
    (5.0, 7.5, "Every pilgrim needs a marble saddle today"),
    (9.0, 11.5, "The glacier violin sounds like thunder again"),
]


@pytest.fixture
def heard(monkeypatch: pytest.MonkeyPatch) -> list[Word]:
    """What the scripted "Whisper" hears; tests fill it in."""
    words: list[Word] = []
    monkeypatch.setattr(Pipeline, "_load_transcriber", lambda self, choice: FakeTranscriber(words))
    monkeypatch.setattr("video_beep_remover.pipeline.silero_speech", lambda audio, on_progress=None: [])
    return words


def movie(tmp_path: Path) -> Path:
    source = make_clip(tmp_path / "movie.mkv", duration=13.0, subtitles=srt(*LINES))
    source.with_suffix(".srt").unlink()
    (tmp_path / "movie.fr.srt").write_text(srt(*LINES), "utf-8")  # wrong language: filtered out
    (tmp_path / "movie.en.sdh.srt").write_text(srt(*LINES), "utf-8")
    return source


def test_subs_lists_candidates_without_checking(tmp_path: Path, heard: list[Word]) -> None:
    result = runner.invoke(app, ["subs", str(movie(tmp_path)), "--no-sync"])
    assert result.exit_code == 0, result.output
    assert "embedded" in result.output and "movie.en.sdh.srt" in result.output
    assert "movie.fr.srt" not in result.output
    assert "no API key" in result.output  # the OpenSubtitles note


def test_subs_checks_sync_and_saves_the_chosen_file(tmp_path: Path, heard: list[Word]) -> None:
    for start, end, text in LINES:
        heard += say(text, start + 0.1, end - 0.2)
    target = tmp_path / "chosen.vtt"
    result = runner.invoke(app, ["subs", str(movie(tmp_path)), "--save", str(target)])
    assert result.exit_code == 0, result.output
    assert "✔" in result.output and "used" in result.output
    assert target.read_text("utf-8").startswith("WEBVTT")  # converted to the format the name asks for
    assert "copper lantern" in target.read_text("utf-8")


def test_subs_fails_when_nothing_is_usable(tmp_path: Path, heard: list[Word]) -> None:
    result = runner.invoke(app, ["subs", str(movie(tmp_path))])
    assert result.exit_code == 1
    assert result.output.count("✘") == 2  # the embedded stream and the English sidecar


def test_subtitles_option_needs_a_single_input(tmp_path: Path) -> None:
    first, second = make_clip(tmp_path / "a.mkv", duration=1.0), make_clip(tmp_path / "b.mkv", duration=1.0)
    subs = tmp_path / "x.srt"
    subs.write_text(srt(*LINES), "utf-8")
    result = runner.invoke(app, ["scan", str(first), str(second), "--subtitles", str(subs)])
    assert result.exit_code == 2
    assert "--subtitles works with a single input" in result.output
