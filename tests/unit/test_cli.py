import io
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from video_beep_remover import __version__
from video_beep_remover.batch import collect_inputs
from video_beep_remover.cli import app
from video_beep_remover.cli.console import ConsoleUI, print_result, utf8_streams
from video_beep_remover.config.schema import Config
from video_beep_remover.errors import UsageError
from video_beep_remover.pipeline import FileResult

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_init_writes_defaults_and_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "conf" / "vbr.toml"
    assert runner.invoke(app, ["config", "init", str(target)]).exit_code == 0
    assert tomllib.loads(target.read_text("utf-8"))["analysis"]["strategy"] == "hybrid"
    assert runner.invoke(app, ["config", "init", str(target)]).exit_code == 2
    assert runner.invoke(app, ["config", "init", str(target), "--force"]).exit_code == 0


def test_config_show_redacts_secrets(tmp_path: Path) -> None:
    config = tmp_path / "vbr.toml"
    config.write_text('[subtitles.opensubtitles]\napi_key = "super-secret"\n', "utf-8")
    result = runner.invoke(app, ["config", "show", "-c", str(config)])
    assert result.exit_code == 0
    assert "super-secret" not in result.stdout
    assert 'api_key = "***"' in result.stdout


def test_config_check_reports_bad_keys(tmp_path: Path) -> None:
    config = tmp_path / "vbr.toml"
    config.write_text("[censor]\nfade = 3\n", "utf-8")
    result = runner.invoke(app, ["config", "check", "-c", str(config)])
    assert result.exit_code == 2
    assert "censor.fade" in result.output


def test_config_check_accepts_defaults() -> None:
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 0
    assert "configuration is valid" in result.output


def test_missing_input_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["clean", str(tmp_path / "missing.mkv")])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_collect_inputs_finds_videos_in_folders(tmp_path: Path) -> None:
    for name in ("b.mkv", "a.MP4", "notes.txt", "c.clean.partial.mkv", "sub/d.mkv"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    assert [i.path.name for i in collect_inputs([tmp_path], recursive=False)] == ["a.MP4", "b.mkv"]
    found = collect_inputs([tmp_path, tmp_path / "b.mkv"], recursive=True)
    assert [(i.path.name, i.from_folder) for i in found] == [
        ("a.MP4", True),
        ("b.mkv", False),
        ("d.mkv", True),
    ]
    try:
        collect_inputs([tmp_path / "sub" / "none"], recursive=False)
    except UsageError as exc:
        assert "not found" in str(exc)


def _config(tmp_path: Path, **overrides: object) -> Config:
    from video_beep_remover.config.loader import load_config

    return load_config(None, env={}, cwd=tmp_path, overrides=dict(overrides)).config


@respx.mock
def test_doctor_checks_the_opensubtitles_key(tmp_path: Path) -> None:
    from video_beep_remover.cli.doctor import opensubtitles_status
    from video_beep_remover.subtitles.opensubtitles import API

    assert opensubtitles_status(_config(tmp_path))[1:] == (
        None,
        "no API key: online subtitle search is skipped (set OPENSUBTITLES_API_KEY)",
    )
    key = {"subtitles.opensubtitles.api_key": "k"}
    assert opensubtitles_status(_config(tmp_path, offline=True, **key))[1] is None  # not checked offline
    search = respx.get(f"{API}/subtitles").mock(return_value=httpx.Response(200, json={"data": []}))
    assert opensubtitles_status(_config(tmp_path, **key))[1:] == (True, "API key accepted")
    assert search.calls.last.request.headers["Api-Key"] == "k"
    respx.post(f"{API}/login").mock(
        return_value=httpx.Response(200, json={"token": "t", "user": {"allowed_downloads": 20}})
    )
    logged_in = _config(
        tmp_path,
        **key,
        **{"subtitles.opensubtitles.username": "me", "subtitles.opensubtitles.password": "pw"},
    )
    assert opensubtitles_status(logged_in)[1:] == (
        True,
        "API key accepted; logged in as me (20 downloads a day)",
    )
    search.mock(return_value=httpx.Response(403, json={"message": "You cannot consume this service"}))
    ok, details = opensubtitles_status(_config(tmp_path, **key))[1:]
    assert ok is False and "rejected" in details


def test_context_flag_turns_the_layer_on() -> None:
    from video_beep_remover.cli.options import overrides

    assert overrides(context=True) == {"context.enabled": True}
    assert overrides(context=False) == {}
    assert overrides(replace=True) == {"replace.enabled": True}


def test_backup_and_in_place_set_the_output_mode() -> None:
    from video_beep_remover.cli.options import overrides
    from video_beep_remover.errors import UsageError

    assert overrides(backup=True) == {"output.mode": "backup"}
    assert overrides(in_place=True) == {"output.mode": "in_place"}
    with pytest.raises(UsageError, match="cannot be combined"):
        overrides(backup=True, in_place=True)


def test_doctor_shows_voice_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from video_beep_remover.cli.doctor import voice_status

    installed = {"torch": False, "torchaudio": False, "f5_tts": False, "demucs": False, "speechbrain": False}
    real = importlib.util.find_spec

    def find_spec(name: str, *rest: Any) -> Any:
        return (object() if installed[name] else None) if name in installed else real(name, *rest)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert voice_status(_config(tmp_path))[1] is None  # off, and not installed: nothing wrong
    ok, details = voice_status(_config(tmp_path, **{"replace.enabled": True}))[1:]
    assert ok is False and "video-beep-remover[voice]" in details
    installed.update(dict.fromkeys(installed, True))
    ok, details = voice_status(_config(tmp_path, **{"replace.enabled": True}))[1:]
    assert ok is True and "F5TTS_v1_Base" in details and "non-commercial" in details


def test_doctor_shows_context_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from video_beep_remover.cli.doctor import context_status

    installed = {"torch": False, "transformers": False}
    real = importlib.util.find_spec

    def find_spec(name: str, *rest: Any) -> Any:
        return (object() if installed[name] else None) if name in installed else real(name, *rest)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    downloaded: set[str] = set()
    hub = SimpleNamespace(try_to_load_from_cache=lambda repo, name: "x" if repo in downloaded else None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cpu = {"transcription.device": "cpu"}
    on = {**cpu, "context.enabled": True}

    assert context_status(_config(tmp_path, **cpu))[1] is None  # off, and not installed: nothing wrong
    ok, details = context_status(_config(tmp_path, **on))[1:]
    assert ok is False and "video-beep-remover[context]" in details

    installed.update(torch=True, transformers=True)
    ok, details = context_status(_config(tmp_path, **on))[1:]
    assert ok is True and "no judge (no GPU); report only" in details
    acting = {**on, "context.harmless": "keep", "context.sexual": "mute"}
    assert (
        "keeps harmless uses and mutes sexual lines (experimental)"
        in context_status(_config(tmp_path, **acting))[2]
    )
    assert "not downloaded yet: unitary/unbiased-toxic-roberta" in details
    judged = {**on, "context.judge": "Qwen/Qwen3-4B-Instruct-2507", "offline": True}
    ok, details = context_status(_config(tmp_path, **judged))[1:]
    assert ok is False and "judge Qwen/Qwen3-4B-Instruct-2507" in details  # offline, nothing downloaded
    downloaded.update({"unitary/unbiased-toxic-roberta", "Qwen/Qwen3-4B-Instruct-2507"})
    ok, details = context_status(_config(tmp_path, **judged))[1:]
    assert ok is True and "not downloaded" not in details

    # Whisper sees a GPU that a CPU-only build of PyTorch cannot use.
    cpu_only = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", cpu_only)
    monkeypatch.setattr("video_beep_remover.asr.faster_whisper.cuda_available", lambda: True)
    details = context_status(_config(tmp_path, **{"context.enabled": True}))[2]
    assert "no judge (no GPU)" in details and "PyTorch cannot use the GPU that Whisper uses" in details

    # A GPU too small for the default judge.
    laptop = SimpleNamespace(
        is_available=lambda: True, get_device_properties=lambda index: SimpleNamespace(total_memory=6 << 30)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=laptop))
    details = context_status(_config(tmp_path, **{"context.enabled": True}))[2]
    assert "no judge (the GPU has 6 GB; the default judge needs about 9)" in details


def test_cache_info_and_clear(tmp_path: Path) -> None:
    config = tmp_path / "vbr.toml"
    config.write_text(f'[cache]\ndir = "{(tmp_path / "c").as_posix()}"\n', "utf-8")
    from video_beep_remover.asr.cache import Transcript, TranscriptCache
    from video_beep_remover.models import Word
    from video_beep_remover.subtitles.cache import SubtitleCache

    SubtitleCache(tmp_path / "c").store(5, b"x" * 2048)
    store = TranscriptCache(tmp_path / "c").store("abc-1", 1, "small.en", "k", 60.0)
    store.add_window(1.0, 5.0, Transcript(1.0, 5.0, (Word("hi", 2.0, 2.2),)))
    (tmp_path / "c" / "context").mkdir()
    (tmp_path / "c" / "context" / "judge-v1.jsonl").write_text('{"key": "k", "answer": "{}"}\n', "utf-8")
    info = runner.invoke(app, ["cache", "info", "-c", str(config)])
    assert info.exit_code == 0 and "subtitles: 1 files, 2 KiB" in info.output
    assert "transcripts: 1 files, 0.0 MiB of at most 5 GB" in info.output
    assert "context judge answers: 1 files" in info.output
    only = runner.invoke(app, ["cache", "clear", "--transcripts", "-c", str(config)])
    assert (
        only.exit_code == 0 and "removed 1 transcript files" in only.output and "subtitle" not in only.output
    )
    cleared = runner.invoke(app, ["cache", "clear", "-c", str(config)])
    assert cleared.exit_code == 0 and "removed 1 downloaded subtitle files" in cleared.output
    assert "removed 0 transcript files" in cleared.output
    assert "removed 1 files of context judge answers" in cleared.output
    assert not list((tmp_path / "c" / "context").glob("*.jsonl"))


def test_results_go_to_stdout_and_messages_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    result = FileResult(
        Path("movie.mkv"),
        "scanned",
        report=Path("movie.vbr.json"),
        detections=2,
        intervals=2,
        strategy="full",
    )
    ConsoleUI().warn("a message")
    print_result(ConsoleUI(), result)
    out, err = capsys.readouterr()
    assert "✔ movie.mkv: 2 listed words, 2 spans to mute (full)" in out and "report: movie.vbr.json" in out
    assert "a message" in err and "movie.mkv" not in err
    print_result(ConsoleUI(quiet=True), result)
    assert "report:" not in capsys.readouterr().out  # --quiet: the result alone


def test_redirected_output_is_written_in_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows writes a redirected stream in its ANSI code page, which has no "→" or "✔"."""
    written = [io.BytesIO(), io.BytesIO()]
    stdout, stderr = (io.TextIOWrapper(raw, encoding="cp1252") for raw in written)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    utf8_streams()
    for stream in (stdout, stderr):
        stream.write("✔ muted 2 spans → movie.clean.mkv")
        stream.flush()
    assert [raw.getvalue().decode("utf-8") for raw in written] == ["✔ muted 2 spans → movie.clean.mkv"] * 2
