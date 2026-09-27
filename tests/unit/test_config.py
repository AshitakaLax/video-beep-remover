import tomllib
from pathlib import Path

import pytest

from video_beep_remover.config import load_config, redact, to_toml
from video_beep_remover.config.loader import defaults_text, find_config_file
from video_beep_remover.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]


def test_packaged_defaults_are_the_documented_example() -> None:
    assert defaults_text() == (REPO / "docs" / "vbr.example.toml").read_text("utf-8")


def test_defaults_load_without_a_config_file(tmp_path: Path) -> None:
    loaded = load_config(env={}, cwd=tmp_path)
    assert loaded.source is None
    cfg = loaded.config
    assert cfg.analysis.strategy == "hybrid"
    assert cfg.censor.fade_ms == 10
    assert list(cfg.lexicon.categories) == ["strong", "mild", "religious", "slurs", "sexual"]
    assert not cfg.lexicon.categories["sexual"].enabled  # phrases; context analysis reports them anyway
    assert not cfg.context.enabled and cfg.context.judge == "auto"


def test_discovery_order(tmp_path: Path) -> None:
    explicit, from_env, local = (tmp_path / name for name in ("explicit.toml", "env.toml", "vbr.toml"))
    for path in (explicit, from_env, local):
        path.write_text("", "utf-8")
    env = {"VBR_CONFIG": str(from_env)}
    assert find_config_file(explicit, cwd=tmp_path, env=env) == explicit
    assert find_config_file(None, cwd=tmp_path, env=env) == from_env
    assert find_config_file(None, cwd=tmp_path, env={}) == local
    assert find_config_file(None, cwd=tmp_path / "missing", env={}) is None


def test_missing_config_files_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml", env={})
    with pytest.raises(ConfigError, match="VBR_CONFIG"):
        load_config(env={"VBR_CONFIG": str(tmp_path / "nope.toml")}, cwd=tmp_path)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "vbr.toml"
    path.write_text(text, "utf-8")
    return path


def test_user_file_is_merged_over_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "[censor]\npad_before_ms = 50\n"), env={}).config
    assert cfg.censor.pad_before_ms == 50
    assert cfg.censor.pad_after_ms == 200
    assert "strong" in cfg.lexicon.categories


def test_user_categories_replace_the_packaged_word_list(tmp_path: Path) -> None:
    path = write(tmp_path, '[lexicon.categories.mine]\nterms = ["frick"]\n')
    cfg = load_config(path, env={}).config
    assert list(cfg.lexicon.categories) == ["mine"]
    assert cfg.lexicon.allow == ["bastardiz*", "wankel*"]  # the rest of [lexicon] still merges


def test_environment_variables_expand(tmp_path: Path) -> None:
    path = write(tmp_path, '[subtitles.opensubtitles]\napi_key = "${MY_KEY}"\nusername = "${UNSET}"\n')
    cfg = load_config(path, env={"MY_KEY": "abc"}).config
    assert cfg.subtitles.opensubtitles.api_key == "abc"
    assert cfg.subtitles.opensubtitles.username == ""


def test_overrides_win(tmp_path: Path) -> None:
    path = write(tmp_path, '[analysis]\nstrategy = "targeted"\n')
    cfg = load_config(path, env={}, overrides={"analysis.strategy": "full", "offline": True}).config
    assert cfg.analysis.strategy == "full"
    assert cfg.offline is True


def test_misspelled_keys_report_their_path(tmp_path: Path) -> None:
    path = write(tmp_path, "[censor]\npad_befor_ms = 5\n")
    with pytest.raises(ConfigError, match=r"censor\.pad_befor_ms"):
        load_config(path, env={})


def test_bad_values_report_their_path(tmp_path: Path) -> None:
    path = write(tmp_path, '[analysis]\nstrategy = "fast"\n')
    with pytest.raises(ConfigError, match=r"analysis\.strategy"):
        load_config(path, env={})


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"vbr\.toml"):
        load_config(write(tmp_path, "[censor\n"), env={})


def test_to_toml_round_trips_the_effective_config(tmp_path: Path) -> None:
    data = load_config(env={}, cwd=tmp_path).config.model_dump(mode="json")
    assert tomllib.loads(to_toml(data)) == data


def test_redact_hides_only_set_secrets() -> None:
    data = {"subtitles": {"opensubtitles": {"api_key": "secret", "password": "", "user_agent": "x"}}}
    assert redact(data) == {
        "subtitles": {"opensubtitles": {"api_key": "***", "password": "", "user_agent": "x"}}
    }
