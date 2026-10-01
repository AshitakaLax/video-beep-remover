"""Where a run's files go (DESIGN.md §6.12): the cleaned video, the original's backup, the report, the
EDL and the review subtitles.

output.mode decides where the cleaned video goes:

- "new" (the default): a new file at output.path (or -o), next to the original, which is left alone;
- "backup" (--backup): the original's own path, the original being kept at output.backup_path;
- "in_place" (--in-place): the original's own path, replacing the original.

The renderer writes <name>.partial<ext> and verifies it before anything is moved, so a failed run leaves
the original as it was (media/render.py)."""

import contextlib
from dataclasses import dataclass
from pathlib import Path

from video_beep_remover.config.schema import OutputConfig
from video_beep_remover.errors import ConfigError, UsageError

MODE_FLAGS = {"backup": "--backup", "in_place": "--in-place"}
REVIEW_SUFFIX = ".review.srt"


def _name(template: str, source: Path, key: str) -> Path:
    try:
        name = template.format(stem=source.stem, ext=source.suffix, dir=str(source.parent))
    except (KeyError, IndexError, ValueError) as exc:
        raise ConfigError(
            f"output.{key} {template!r}: unknown placeholder {exc} (use {{stem}}, {{ext}}, {{dir}})"
        ) from exc
    return Path(name).expanduser()


def resolve_output(source: Path, template: str, explicit: Path | None, *, many: bool) -> Path:
    """Apply output.path ({stem}, {ext}, {dir}); -o may name a file or a directory.

    An -o path without a file extension is a directory, even if it does not exist yet."""
    candidate = _name(template, source, "path")
    if explicit is not None:
        if many or explicit.is_dir() or not explicit.suffix or str(explicit).endswith(("/", "\\")):
            return explicit / candidate.name
        return explicit
    return candidate if candidate.is_absolute() else source.parent / candidate


def backup_path(source: Path, template: str) -> Path:
    """Where "backup" mode keeps the original (output.backup_path)."""
    candidate = _name(template, source, "backup_path")
    return candidate if candidate.is_absolute() else source.parent / candidate


@dataclass(frozen=True)
class Placement:
    source: Path
    output: Path  # where the cleaned video goes
    backup: Path | None = None  # "backup" mode: where the original goes, just before the output replaces it

    @property
    def replaces_source(self) -> bool:
        return self.output == self.source


def place(source: Path, config: OutputConfig, explicit: Path | None, *, many: bool) -> Placement:
    """Where the cleaned video of `source` goes, following output.mode; `explicit` is -o."""
    if config.mode == "new":
        output = resolve_output(source, config.path, explicit, many=many)
        if output.resolve() == source.resolve():
            raise UsageError(
                f"the output would overwrite the input: {output} (--in-place or --backup replace it)"
            )
        return Placement(source, output)
    if explicit is not None:
        flag = MODE_FLAGS[config.mode]
        raise UsageError(f"-o cannot be combined with {flag}: the cleaned file takes the original's place")
    if config.mode == "in_place":
        return Placement(source, source)
    backup = backup_path(source, config.backup_path)
    if backup.resolve() == source.resolve():
        raise ConfigError(f"output.backup_path {config.backup_path!r} names the original itself")
    return Placement(source, source, backup)


def report_path(explicit: Path | None, source: Path, output: Path | None, *, many: bool) -> Path:
    """Where the JSON report goes: --report (a file, or a directory for several inputs), or next to the
    output (the input for a scan), named <stem>.vbr.json."""
    if explicit is not None:
        if many or explicit.is_dir():
            return explicit / f"{source.stem}.vbr.json"
        return explicit
    base = output or source
    return base.with_name(f"{base.stem}.vbr.json")


def find_report(video: Path, explicit: Path | None, config: OutputConfig, *, many: bool) -> Path | None:
    """The report `vbr render` reads for `video`: --report (a file, or a directory for several videos), or
    the one a scan or a clean of it wrote. None if there is none."""
    if explicit is not None:
        return report_path(explicit, video, None, many=many)
    candidates = [report_path(None, video, None, many=False)]  # vbr scan, or a clean in place
    with contextlib.suppress(ConfigError):
        output = resolve_output(video, config.path, None, many=False)
        candidates.append(report_path(None, video, output, many=False))  # a clean to a new file
    return next((c for c in candidates if c.is_file()), None)


def edl_path(source: Path, backup: Path | None) -> Path:
    """The EDL mutes the unmodified original in Kodi or MPlayer, so it goes next to it."""
    return (backup or source).with_suffix(".edl")


def review_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}{REVIEW_SUFFIX}")
