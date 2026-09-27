"""Configuration (DESIGN.md §4): the pydantic schema, the packaged defaults, and loading a user's file
with environment variables and command-line overrides."""

from video_beep_remover.config.loader import (
    LoadedConfig,
    cache_root,
    config_hash,
    defaults_text,
    load_config,
    redact,
    to_toml,
    user_cache_path,
    user_config_path,
)
from video_beep_remover.config.schema import Config

__all__ = [
    "Config",
    "LoadedConfig",
    "cache_root",
    "config_hash",
    "defaults_text",
    "load_config",
    "redact",
    "to_toml",
    "user_cache_path",
    "user_config_path",
]
