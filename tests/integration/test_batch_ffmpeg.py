"""Batch runs over folders: vbr's own outputs are skipped, and renders overlap the next analysis."""

import shutil
import threading
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import FakeTranscriber, StrictUI, make_clip, words
from video_beep_remover.batch import Input, Outcome, collect_inputs, output_clashes, run_batch, skip_outputs
from video_beep_remover.cli import app
from video_beep_remover.config import load_config
from video_beep_remover.errors import MediaError
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe
from video_beep_remover.pipeline import Pipeline, RunOptions

pytestmark = pytest.mark.ffmpeg
SPOKEN = words(("well", 1.0, 1.3), ("damn", 2.0, 2.4))


def pipeline(tmp_path: Path) -> Pipeline:
    loaded = load_config(None, env={}, cwd=tmp_path, overrides={"transcription.device": "cpu"})
    fake = FakeTranscriber(SPOKEN)
    return Pipeline(loaded, ui=StrictUI(), transcriber_factory=lambda choice: fake)


def test_outputs_found_in_folders_are_skipped(tmp_path: Path) -> None:
    inputs = [Input(tmp_path / name, True) for name in ("a.mkv", "a.clean.mkv", "b.mp4")]
    kept, skipped = skip_outputs(inputs, "{stem}.clean{ext}", None)
    assert [i.path.name for i in kept] == ["a.mkv", "b.mp4"]
    assert [(r.input.name, r.status, r.notes) for r in skipped] == [
        ("a.clean.mkv", "skipped", ["it is the output for a.mkv"])
    ]
    named = [Input(tmp_path / "a.mkv", True), Input(tmp_path / "a.clean.mkv", False)]
    assert skip_outputs(named, "{stem}.clean{ext}", None)[1] == []  # named on the command line: kept


def test_inputs_with_the_same_output_are_caught_before_rendering(tmp_path: Path) -> None:
    inputs = [
        Input(tmp_path / "s1" / "Episode 01.mkv", True),
        Input(tmp_path / "s2" / "Episode 01.mkv", True),
    ]
    out = tmp_path / "clean"
    writers, clashes = output_clashes(inputs, "{stem}.clean{ext}", out, many=True)
    assert writers == {(out / "Episode 01.clean.mkv").resolve(): inputs[0].path}
    assert list(clashes) == [inputs[1].path] and "would also be written for" in clashes[inputs[1].path]
    assert output_clashes(inputs, "{stem}.clean{ext}", None, many=True)[1] == {}  # next to each input


def test_a_colliding_input_fails_and_the_first_output_survives(tmp_path: Path) -> None:
    for season in ("s1", "s2"):
        (tmp_path / season).mkdir()
        make_clip(tmp_path / season / "Episode 01.mkv", duration=3.0)
    inputs = collect_inputs([tmp_path / "s1", tmp_path / "s2"], recursive=False)
    outcomes: list[Outcome] = []
    options = RunOptions(output=tmp_path / "clean", overwrite=True)  # not even --overwrite lets it through
    run_batch(pipeline(tmp_path), inputs, options, outcomes.append)
    assert [
        (o.path.parent.name, o.result.status if o.result else type(o.error).__name__) for o in outcomes
    ] == [
        ("s1", "cleaned"),
        ("s2", "UsageError"),
    ]
    assert (tmp_path / "clean" / "Episode 01.clean.mkv").is_file()


def test_batch_renders_in_the_background_and_reports_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "library"
    folder.mkdir()
    for name in ("a.mkv", "b.mkv", "c.mkv"):
        make_clip(folder / name, duration=4.0)
    run = pipeline(tmp_path)
    run.process(folder / "a.mkv", RunOptions())  # an earlier run's output, a.clean.mkv, is in the folder
    shutil.copy(folder / "a.clean.mkv", folder / "0-copy.mkv")  # a vbr output under another name
    (folder / "a.clean.vbr.json").unlink()

    threads: dict[str, str] = {}
    original = Pipeline._render

    def render(self: Pipeline, job: Any, ui: Any) -> None:
        threads[job.source.name] = threading.current_thread().name
        original(self, job, ui)

    monkeypatch.setattr(Pipeline, "_render", render)
    inputs, skipped = skip_outputs(collect_inputs([folder], recursive=False), "{stem}.clean{ext}", None)
    assert [r.input.name for r in skipped] == ["a.clean.mkv"]
    outcomes: list[Outcome] = []
    run_batch(run, inputs, RunOptions(overwrite=True), outcomes.append)

    assert [(o.path.name, o.result.status if o.result else o.error) for o in outcomes] == [
        ("0-copy.mkv", "skipped"),
        ("a.mkv", "cleaned"),
        ("b.mkv", "cleaned"),
        ("c.mkv", "cleaned"),
    ]
    assert outcomes[0].result is not None and "already censored by vbr" in outcomes[0].result.notes[0]
    assert threads["a.mkv"].startswith("vbr-render") and threads["b.mkv"].startswith("vbr-render")
    assert threads["c.mkv"] == "MainThread"  # nothing left to overlap: rendered with a progress bar
    tag = probe(FFmpeg(), folder / "b.clean.mkv").tags["vbr_censored"]
    assert tag == run.tag and tag.count(";") == 1


def test_a_failed_background_render_fails_only_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.mkv", "b.mkv"):
        make_clip(tmp_path / name, duration=3.0)
    original = Pipeline._render

    def render(self: Pipeline, job: Any, ui: Any) -> None:
        if job.source.name == "a.mkv":
            ui.warn("about to fail")
            raise MediaError("disk full")
        original(self, job, ui)

    monkeypatch.setattr(Pipeline, "_render", render)
    outcomes: list[Outcome] = []
    inputs = [Input(tmp_path / "a.mkv", True), Input(tmp_path / "b.mkv", True)]
    run_batch(pipeline(tmp_path), inputs, RunOptions(), outcomes.append)
    assert [(o.path.name, str(o.error) if o.error else o.result and o.result.status) for o in outcomes] == [
        ("a.mkv", "disk full"),
        ("b.mkv", "cleaned"),
    ]
    assert outcomes[0].messages == [("warn", "about to fail")]
    assert not list(tmp_path.glob("a.clean*"))


def test_clean_a_folder_from_the_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Pipeline, "_load_faster_whisper", lambda self, choice: FakeTranscriber(SPOKEN))
    monkeypatch.setattr("video_beep_remover.pipeline.silero_speech", lambda audio, on_progress=None: [])
    for name in ("a.mkv", "b.mkv"):
        make_clip(tmp_path / name, duration=3.0)
    first = runner.invoke(app, ["clean", str(tmp_path), "--device", "cpu"])
    assert first.exit_code == 0, first.output
    assert first.output.index("a.mkv: muted 1 spans") < first.output.index("b.mkv: muted 1 spans")
    second = runner.invoke(app, ["clean", str(tmp_path), "--device", "cpu", "--skip-existing"])
    assert second.exit_code == 0, second.output
    assert "a.clean.mkv: skipped (it is the output for a.mkv)" in second.output
    assert "a.mkv: skipped (output already exists)" in second.output
    assert not list(tmp_path.glob("*.clean.clean.*"))


runner = CliRunner()
