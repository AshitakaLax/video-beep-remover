"""The `vbr` command line (DESIGN.md §3). One module per group of commands:

- videos.py: clean, scan and render, which process video files or folders of them;
- subs.py: subs; doctor.py: doctor; configure.py: config; cache.py: cache.

app.py holds the Typer application, console.py the console output, and options.py the options the
commands share and the configuration keys they set."""

from video_beep_remover.cli import (  # noqa: F401  (registers the commands)
    cache,
    configure,
    doctor,
    subs,
    videos,
)
from video_beep_remover.cli.app import app

__all__ = ["app", "main"]


def main() -> None:
    app()
