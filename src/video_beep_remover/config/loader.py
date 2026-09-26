"""Find, merge and validate the configuration (DESIGN.md §4.1)."""

import copy
import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import platformdirs
from pydantic import ValidationError

from video_beep_remover.config.schema import Config
from video_beep_remover.errors import ConfigError

APP_NAME = "video-beep-remover"
ENV_CONFIG = "VBR_CONFIG"
LOCAL_CONFIG = "vbr.toml"
SECRET_KEYS = frozenset({"api_key", "password"})

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class LoadedConfig:
    config: Config
    source: Path | None  # the file that was merged over the defaults, if any
    data: dict[str, Any]  # the merged, expanded dictionary that was validated

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths in the config (such as word files) resolve against."""
        return self.source.parent if self.source else Path.cwd()


def user_config_path() -> Path:
    return platformdirs.user_config_path(APP_NAME, appauthor=False) / "config.toml"


def user_cache_path() -> Path:
    return platformdirs.user_cache_path(APP_NAME, appauthor=False)


def cache_root(config: Config) -> Path:
    """cache.dir, where "auto" means the per-user cache dir (e.g. ~/.cache/video-beep-remover)."""
    return user_cache_path() if config.cache.dir == "auto" else Path(config.cache.dir).expanduser()


def defaults_text() -> str:
    return resources.files("video_beep_remover.config").joinpath("defaults.toml").read_text("utf-8")


def find_config_file(explicit: Path | None, *, cwd: Path, env: Mapping[str, str]) -> Path | None:
    """--config, then $VBR_CONFIG, then ./vbr.toml, then the per-user config file."""
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    from_env = env.get(ENV_CONFIG)
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise ConfigError(f"${ENV_CONFIG} points to a missing file: {path}")
        return path
    local = cwd / LOCAL_CONFIG
    if local.is_file():
        return local
    user = user_config_path()
    return user if user.is_file() else None


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge tables recursively; any other value (including lists) in `override` replaces the base."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_user_config(defaults: Mapping[str, Any], user: Mapping[str, Any]) -> dict[str, Any]:
    merged = deep_merge(defaults, user)
    lexicon = user.get("lexicon")
    if isinstance(lexicon, Mapping) and isinstance(lexicon.get("categories"), Mapping):
        # Categories the user defines replace the packaged word list instead of adding to it.
        merged["lexicon"]["categories"] = dict(lexicon["categories"])
    return merged


def expand_env(value: Any, env: Mapping[str, str]) -> Any:
    """Replace ${NAME} in every string with the environment variable (unset -> empty string)."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: env.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env(item, env) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item, env) for key, item in value.items()}
    return value


def apply_overrides(data: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Set dotted keys such as "analysis.strategy" (command-line flags win over every file)."""
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[leaf] = value


def format_validation_error(error: ValidationError, source: Path | None) -> str:
    where = f" in {source}" if source else ""
    lines = [f"invalid configuration{where}:"]
    seen: set[str] = set()
    for item in error.errors():
        loc = ".".join(str(part) for part in item["loc"])
        line = f"  {loc}: {item['msg']}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def load_config(
    explicit: Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> LoadedConfig:
    env = os.environ if env is None else env
    defaults = tomllib.loads(defaults_text())
    source = find_config_file(explicit, cwd=cwd or Path.cwd(), env=env)
    data: dict[str, Any] = defaults
    if source is not None:
        try:
            user = tomllib.loads(source.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{source}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read {source}: {exc}") from exc
        data = merge_user_config(defaults, user)
    data = expand_env(copy.deepcopy(data), env)
    if overrides:
        apply_overrides(data, overrides)
    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, source)) from exc
    return LoadedConfig(config=config, source=source, data=data)


def config_hash(config: Config) -> str:
    """A short fingerprint of the settings (secrets left out), recorded in every output's VBR_CENSORED tag."""
    text = json.dumps(redact(config.model_dump(mode="json")), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def redact(data: Any) -> Any:
    """Copy of the config dictionary with non-empty secrets replaced."""
    if isinstance(data, dict):
        return {
            key: ("***" if key in SECRET_KEYS and isinstance(value, str) and value else redact(value))
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [redact(item) for item in data]
    return data


def to_toml(data: Mapping[str, Any]) -> str:
    """Serialize the config dictionary (tables, strings, numbers, booleans, lists) as TOML."""
    lines: list[str] = []

    def key(name: str) -> str:
        return name if _BARE_KEY.match(name) else json.dumps(name, ensure_ascii=False)

    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return "[" + ", ".join(scalar(item) for item in value) + "]"
        raise TypeError(f"cannot serialize {type(value).__name__} as TOML")

    def emit(table: Mapping[str, Any], prefix: str) -> None:
        plain = {k: v for k, v in table.items() if not isinstance(v, Mapping)}
        nested = {k: v for k, v in table.items() if isinstance(v, Mapping)}
        if prefix and (plain or not nested):
            lines.append(f"[{prefix}]")
        for name, value in plain.items():
            lines.append(f"{key(name)} = {scalar(value)}")
        if plain or (prefix and not nested):
            lines.append("")
        for name, value in nested.items():
            emit(value, f"{prefix}.{key(name)}" if prefix else key(name))

    emit(data, "")
    return "\n".join(lines).rstrip() + "\n"
