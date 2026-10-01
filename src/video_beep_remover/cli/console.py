"""Console output shared by the commands: messages, progress bars, logging and per-file summaries."""

import codecs
import contextlib
import io
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from video_beep_remover.config.loader import LoadedConfig, load_config
from video_beep_remover.errors import VbrError
from video_beep_remover.pipeline import FileResult
from video_beep_remover.pipeline import Progress as ProgressFn

console = Console(stderr=True, highlight=False)  # messages and progress
results = Console(highlight=False, soft_wrap=True)  # each file's result, never wrapped


def utf8_streams() -> None:
    """Write UTF-8 when the output goes to a file or a pipe. Windows would otherwise use its ANSI code
    page, which has no "→" or "✔": Python writes "\\u2192" instead, or fails on stdout."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and codecs.lookup(stream.encoding).name != "utf-8":
            stream.reconfigure(encoding="utf-8")


class ConsoleUI:
    def __init__(self, *, quiet: bool = False) -> None:
        self.quiet = quiet

    def info(self, message: str) -> None:
        if not self.quiet:
            console.print(escape(message))

    def warn(self, message: str) -> None:
        console.print(f"[yellow]warning:[/] {escape(message)}")

    def error(self, message: str) -> None:
        console.print(f"[red]error:[/] {escape(message)}")

    @contextlib.contextmanager
    def progress(self, label: str, total: float) -> Iterator[ProgressFn]:
        if self.quiet or total <= 0:
            yield lambda done: None
            return
        columns = (TextColumn(f"{label:<14}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn())
        with Progress(*columns, console=console, transient=True) as bar:
            task = bar.add_task(label, total=total)
            yield lambda done: bar.update(task, completed=min(done, total))

    @contextlib.contextmanager
    def status(self, label: str) -> Iterator[None]:
        if self.quiet:
            yield
            return
        with console.status(escape(label)):
            yield


def setup_logging(verbose: bool) -> None:
    if not verbose:
        # huggingface_hub sets its own level when first used, and prints through a handler of its own.
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Model downloads log an "unauthenticated requests" warning that is only noise for this tool.
    for noisy in ("faster_whisper", "whisperx", "httpx", "urllib3", "huggingface_hub", "speechbrain"):
        logging.getLogger(noisy).setLevel(logging.INFO if verbose else logging.ERROR)


def fail(error: VbrError) -> typer.Exit:
    ConsoleUI().error(str(error))
    return typer.Exit(error.exit_code)


def load_or_exit(config: Path | None) -> LoadedConfig:
    try:
        return load_config(config)
    except VbrError as exc:
        raise fail(exc) from exc


def summarize(result: FileResult) -> str:
    name = result.input.name
    if result.status == "skipped":
        return f"{name}: skipped ({'; '.join(result.notes)})"
    replaced = f", {result.replaced} replaced" if result.replaced else ""
    if result.status == "scanned":
        return (
            f"{name}: {result.detections} listed words, {result.intervals} spans to mute{replaced} "
            f"({result.strategy})"
        )
    if result.status == "clean":
        return f"{name}: nothing to mute, no output written"
    target = "in place" if result.output == result.input else f"→ {result.output}"
    if result.status == "copied":
        return f"{name}: nothing to mute, copied {target} ({result.strategy})"
    how = "from the report" if result.strategy == "report" else result.strategy
    return f"{name}: muted {result.intervals} spans{replaced} {target} ({how})"


def print_result(ui: ConsoleUI, result: FileResult) -> None:
    """A file's result, on stdout, so that it can be redirected apart from the messages on stderr."""
    results.print(f"[green]✔[/] {escape(summarize(result))}")
    if ui.quiet:
        return
    for label, path in (
        ("original kept as", result.backup if result.status != "skipped" else None),
        ("report", result.report),
        ("EDL", result.edl),
        ("review subtitles", result.review),
        ("censored subtitles", result.subtitle_copy),
    ):
        if path:
            results.print(escape(f"  {label}: {path}"))
