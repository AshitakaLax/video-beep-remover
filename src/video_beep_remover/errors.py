"""Exceptions and the exit codes they map to (DESIGN.md §3.3)."""

EXIT_OK = 0
EXIT_PROCESSING = 1
EXIT_USAGE = 2
EXIT_DEPENDENCY = 3
EXIT_PARTIAL = 4


class VbrError(Exception):
    """Base class for errors that end a run with a clear message instead of a traceback."""

    exit_code = EXIT_PROCESSING


class ConfigError(VbrError):
    exit_code = EXIT_USAGE


class UsageError(VbrError):
    exit_code = EXIT_USAGE


class DependencyError(VbrError):
    """FFmpeg, a model or an encoder is missing or unusable."""

    exit_code = EXIT_DEPENDENCY


class MediaError(VbrError):
    """Probing, decoding or encoding a media file failed."""


class RenderError(VbrError):
    """The cleaned file could not be produced or failed verification."""


class SubtitleError(VbrError):
    """A subtitle file could not be read or parsed."""
