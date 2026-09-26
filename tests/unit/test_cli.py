import tomllib
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from video_beep_remover import __version__
from video_beep_remover.batch import collect_inputs
from video_beep_remover.cli import app
from video_beep_remover.config.schema import Config
from video_beep_remover.errors import UsageError

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
    from video_beep_remover.cli import _opensubtitles_status
    from video_beep_remover.subtitles.opensubtitles import API

    assert _opensubtitles_status(_config(tmp_path))[1:] == (
        None,
        "no API key: online subtitle search is skipped (set OPENSUBTITLES_API_KEY)",
    )
    key = {"subtitles.opensubtitles.api_key": "k"}
    assert _opensubtitles_status(_config(tmp_path, offline=True, **key))[1] is None  # not checked offline
    search = respx.get(f"{API}/subtitles").mock(return_value=httpx.Response(200, json={"data": []}))
    assert _opensubtitles_status(_config(tmp_path, **key))[1:] == (True, "API key accepted")
    assert search.calls.last.request.headers["Api-Key"] == "k"
    respx.post(f"{API}/login").mock(
        return_value=httpx.Response(200, json={"token": "t", "user": {"allowed_downloads": 20}})
    )
    logged_in = _config(
        tmp_path,
        **key,
        **{"subtitles.opensubtitles.username": "me", "subtitles.opensubtitles.password": "pw"},
    )
    assert _opensubtitles_status(logged_in)[1:] == (
        True,
        "API key accepted; logged in as me (20 downloads a day)",
    )
    search.mock(return_value=httpx.Response(403, json={"message": "You cannot consume this service"}))
    ok, details = _opensubtitles_status(_config(tmp_path, **key))[1:]
    assert ok is False and "rejected" in details


def test_cache_info_and_clear(tmp_path: Path) -> None:
    config = tmp_path / "vbr.toml"
    config.write_text(f'[cache]\ndir = "{(tmp_path / "c").as_posix()}"\n', "utf-8")
    from video_beep_remover.subtitles.cache import SubtitleCache

    SubtitleCache(tmp_path / "c").store(5, b"x" * 2048)
    info = runner.invoke(app, ["cache", "info", "-c", str(config)])
    assert info.exit_code == 0 and "1 files, 2 KiB" in info.output
    cleared = runner.invoke(app, ["cache", "clear", "-c", str(config)])
    assert cleared.exit_code == 0 and "removed 1 files" in cleared.output
