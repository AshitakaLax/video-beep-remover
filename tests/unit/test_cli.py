import tomllib
from pathlib import Path

from typer.testing import CliRunner

from video_beep_remover import __version__
from video_beep_remover.cli import app, collect_inputs
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
    assert [p.name for p in collect_inputs([tmp_path], recursive=False)] == ["a.MP4", "b.mkv"]
    assert [p.name for p in collect_inputs([tmp_path], recursive=True)] == ["a.MP4", "b.mkv", "d.mkv"]
    try:
        collect_inputs([tmp_path / "sub" / "none"], recursive=False)
    except UsageError as exc:
        assert "not found" in str(exc)
