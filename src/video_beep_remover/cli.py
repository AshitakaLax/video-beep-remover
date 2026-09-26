"""The `vbr` command line (DESIGN.md §3)."""

import contextlib
import logging
import tempfile
from collections.abc import Iterator
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from video_beep_remover import __version__
from video_beep_remover.config.loader import (
    LoadedConfig,
    defaults_text,
    load_config,
    redact,
    to_toml,
    user_config_path,
)
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.errors import (
    EXIT_DEPENDENCY,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_PROCESSING,
    EXIT_USAGE,
    DependencyError,
    UsageError,
    VbrError,
)
from video_beep_remover.media.probe import VIDEO_SUFFIXES
from video_beep_remover.pipeline import FileResult, Pipeline, RunOptions
from video_beep_remover.pipeline import Progress as ProgressFn

app = typer.Typer(
    name="vbr",
    help="Mute a configurable list of words in video files.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Create, show and check configuration files.", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console(stderr=True, highlight=False)


class StrategyChoice(StrEnum):
    hybrid = "hybrid"
    targeted = "targeted"
    full = "full"


class DeviceChoice(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


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


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Model downloads log an "unauthenticated requests" warning that is only noise for this tool.
    for noisy in ("faster_whisper", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.INFO if verbose else logging.ERROR)


def _fail(error: VbrError) -> typer.Exit:
    ConsoleUI().error(str(error))
    return typer.Exit(error.exit_code)


def collect_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = path.rglob("*") if recursive else path.iterdir()
            files += sorted(
                p
                for p in found
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and ".partial." not in p.name
            )
        elif path.is_file():
            files.append(path)
        else:
            raise UsageError(f"not found: {path}")
    unique = list(dict.fromkeys(p.resolve() for p in files))
    if not unique:
        raise UsageError("no video files found")
    return unique


def _overrides(**flags: Any) -> dict[str, Any]:
    keys = {
        "strategy": "analysis.strategy",
        "language": "analysis.language",
        "audio_stream": "analysis.audio_stream",
        "model": "transcription.model",
        "device": "transcription.device",
    }
    overrides = {
        keys[name]: value.value if isinstance(value, Enum) else value
        for name, value in flags.items()
        if name in keys and value is not None
    }
    if flags.get("no_fallback"):
        overrides["analysis.fallback_to_full"] = False
    if flags.get("offline"):
        overrides["offline"] = True
    return overrides


def _summarize(result: FileResult) -> str:
    name = result.input.name
    if result.status == "skipped":
        return f"{name}: skipped ({'; '.join(result.notes)})"
    if result.status == "scanned":
        return (
            f"{name}: {result.detections} listed words, {result.intervals} spans to mute ({result.strategy})"
        )
    if result.status == "clean":
        return f"{name}: nothing to mute, no output written"
    if result.status == "copied":
        return f"{name}: nothing to mute, copied → {result.output} ({result.strategy})"
    return f"{name}: muted {result.intervals} spans → {result.output} ({result.strategy})"


def _run(
    inputs: list[Path],
    config: Path | None,
    options: RunOptions,
    categories: str | None,
    quiet: bool,
    verbose: bool,
    **flags: Any,
) -> None:
    _setup_logging(verbose)
    ui = ConsoleUI(quiet=quiet)
    try:
        loaded = load_config(config, overrides=_overrides(**flags))
        files = collect_inputs(inputs, flags.get("recursive", False))
        many = len(files) > 1
        if many and options.output is not None and options.output.exists() and not options.output.is_dir():
            raise UsageError("with several inputs, --output must be a directory")
        if many and options.subtitles is not None:
            raise UsageError("--subtitles works with a single input")
        cats = categories.split(",") if categories else None
        pipeline = Pipeline(loaded, ui=ui, categories=cats)
    except VbrError as exc:
        raise _fail(exc) from exc

    failures = 0
    for path in files:
        try:
            result = pipeline.process(path, options, many=many)
        except DependencyError as exc:
            raise _fail(exc) from exc  # would fail for every file: stop the batch
        except VbrError as exc:
            if not many:
                raise _fail(exc) from exc
            failures += 1
            ui.error(f"{path.name}: {exc}")
            continue
        console.print(f"[green]✔[/] {escape(_summarize(result))}")
        if result.report:
            ui.info(f"  report: {result.report}")
        if result.edl:
            ui.info(f"  EDL: {result.edl}")
    if failures:
        ui.error(f"{failures} of {len(files)} files failed")
        raise typer.Exit(EXIT_PARTIAL)


InputsArg = Annotated[list[Path], typer.Argument(help="Video files, or folders of them.", show_default=False)]
ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Config file (see `vbr config`).")]
StrategyOpt = Annotated[StrategyChoice | None, typer.Option(help="How to find listed words.")]
SubtitlesOpt = Annotated[
    Path | None, typer.Option("--subtitles", help="Use this subtitle file instead of searching for one.")
]
NoFallbackOpt = Annotated[
    bool, typer.Option("--no-fallback", help="Fail instead of transcribing everything.")
]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="No network access at all.")]
CategoriesOpt = Annotated[str | None, typer.Option(help="Enable exactly these categories, e.g. strong,mild.")]
ModelOpt = Annotated[str | None, typer.Option(help="Whisper model, e.g. large-v3-turbo or small.en.")]
DeviceOpt = Annotated[DeviceChoice | None, typer.Option(help="Where to run Whisper.")]
LanguageOpt = Annotated[str | None, typer.Option(help="Spoken language, e.g. en.")]
AudioStreamOpt = Annotated[int | None, typer.Option(help="ffprobe index of the dialogue audio stream.")]
ReportOpt = Annotated[Path | None, typer.Option(help="Where to write the JSON report.")]
EdlOpt = Annotated[bool, typer.Option("--edl", help="Also write <input>.edl, a mute list for Kodi/MPlayer.")]
RecursiveOpt = Annotated[bool, typer.Option("--recursive", "-r", help="Search folders recursively.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show debug output.")]
QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Only show results and errors.")]


@app.command()
def clean(
    inputs: InputsArg,
    config: ConfigOpt = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file or directory.")] = None,
    strategy: StrategyOpt = None,
    subtitles: SubtitlesOpt = None,
    no_fallback: NoFallbackOpt = False,
    offline: OfflineOpt = False,
    categories: CategoriesOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    language: LanguageOpt = None,
    audio_stream: AudioStreamOpt = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Detect only; same as `vbr scan`.")] = False,
    report: ReportOpt = None,
    edl: EdlOpt = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing outputs.")] = False,
    skip_existing: Annotated[
        bool, typer.Option("--skip-existing", help="Skip inputs whose output exists.")
    ] = False,
    recursive: RecursiveOpt = False,
    keep_temp: Annotated[
        bool, typer.Option("--keep-temp", help="Keep temporary files for debugging.")
    ] = False,
    verbose: VerboseOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Find the listed words and write a copy of each video with them muted."""
    options = RunOptions(
        dry_run=dry_run,
        output=output,
        report=report,
        edl=edl,
        overwrite=overwrite,
        skip_existing=skip_existing,
        keep_temp=keep_temp,
        subtitles=subtitles,
    )
    _run(
        inputs,
        config,
        options,
        categories,
        quiet,
        verbose,
        strategy=strategy,
        no_fallback=no_fallback,
        offline=offline,
        model=model,
        device=device,
        language=language,
        audio_stream=audio_stream,
        recursive=recursive,
    )


@app.command()
def scan(
    inputs: InputsArg,
    config: ConfigOpt = None,
    strategy: StrategyOpt = None,
    subtitles: SubtitlesOpt = None,
    no_fallback: NoFallbackOpt = False,
    offline: OfflineOpt = False,
    categories: CategoriesOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    language: LanguageOpt = None,
    audio_stream: AudioStreamOpt = None,
    report: ReportOpt = None,
    edl: EdlOpt = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an existing EDL.")] = False,
    recursive: RecursiveOpt = False,
    verbose: VerboseOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Find the listed words and write a report (and optional EDL), without writing video."""
    options = RunOptions(dry_run=True, report=report, edl=edl, overwrite=overwrite, subtitles=subtitles)
    _run(
        inputs,
        config,
        options,
        categories,
        quiet,
        verbose,
        strategy=strategy,
        no_fallback=no_fallback,
        offline=offline,
        model=model,
        device=device,
        language=language,
        audio_stream=audio_stream,
        recursive=recursive,
    )


@app.command("subs")
def subs_command(
    video: Annotated[Path, typer.Argument(help="Video file.", show_default=False)],
    config: ConfigOpt = None,
    subtitles: SubtitlesOpt = None,
    offline: OfflineOpt = False,
    language: LanguageOpt = None,
    audio_stream: AudioStreamOpt = None,
    device: DeviceOpt = None,
    no_sync: Annotated[bool, typer.Option("--no-sync", help="Only list the candidates.")] = False,
    save: Annotated[
        Path | None,
        typer.Option("--save", help="Write the chosen subtitles here; the extension sets the format."),
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Show subtitle candidates, their ranking and the sync check. Exits with 1 if none is usable."""
    from video_beep_remover import guided
    from video_beep_remover.media.audio import SeekingAudioSource
    from video_beep_remover.media.probe import probe, select_audio_stream
    from video_beep_remover.subtitles.acquire import SubtitleLoader, find_candidates
    from video_beep_remover.subtitles.save import save_subtitles

    _setup_logging(verbose)
    overrides = _overrides(offline=offline, language=language, audio_stream=audio_stream, device=device)
    try:
        loaded = load_config(config, overrides=overrides)
        cfg = loaded.config
        if not video.is_file():
            raise UsageError(f"not found: {video}")
        if subtitles is not None and not subtitles.is_file():
            raise UsageError(f"subtitle file not found: {subtitles}")
        pipeline = Pipeline(loaded, ui=ConsoleUI(quiet=True))
        info = probe(pipeline.ff, video)
        search = find_candidates(info, cfg.subtitles, offline=cfg.offline, explicit=subtitles)
        table = Table(show_header=True, header_style="bold")
        for column in ("#", "source", "subtitles", "language", "SDH", "score"):
            table.add_column(column)
        for number, candidate in enumerate(search.candidates, 1):
            table.add_row(
                str(number),
                candidate.source,
                escape(candidate.label),
                candidate.language or "?",
                "yes" if candidate.hearing_impaired else "",
                f"{candidate.score:.0f}" if candidate.source != "explicit" else "",
            )
        for note in search.notes:
            console.print(f"[dim]{escape(note)}[/]")
        if not search.candidates:
            console.print("no subtitle candidates found")
            raise typer.Exit(EXIT_PROCESSING)
        console.print(table)

        with tempfile.TemporaryDirectory(prefix="vbr-") as tmp:
            workdir = Path(tmp)
            if no_sync:
                top = search.candidates[0]
                text = SubtitleLoader(pipeline.ff, video, workdir, [top]).text(top)
            else:
                stream = select_audio_stream(info, cfg.analysis.language, cfg.analysis.audio_stream)
                audio = SeekingAudioSource(pipeline.ff, video, stream.index)
                with console.status("Checking the sync (transcribes a few seconds around six cues)"):
                    selection = guided.select_subtitles(
                        pipeline.guided_context(), info, audio, workdir, subtitles
                    )
                for entry in selection.tried:
                    used = entry.get("result") == "used"
                    sync = entry.get("sync")
                    detail = ""
                    if sync:
                        fidelity = "unknown" if sync["fidelity"] is None else f"{sync['fidelity']:.2f}"
                        detail = (
                            f" [{sync['matched']}/{sync['anchors']} anchors, offset {sync['offset']:+.2f} s, "
                            f"error {sync['error']:.2f} s, fidelity {fidelity}]"
                        )
                    mark = "[green]✔[/]" if used else "[red]✘[/]"
                    console.print(
                        f"{mark} {escape(entry['label'])}: {escape(str(entry['result']))}{escape(detail)}"
                    )
                if selection.chosen is None:
                    raise typer.Exit(EXIT_PROCESSING)
                text = selection.chosen.text
            if save is not None:
                save_subtitles(text, save, fps=info.frame_rate)
                console.print(f"Wrote {escape(str(save))}")
    except VbrError as exc:
        raise _fail(exc) from exc


@config_app.command("init")
def config_init(
    path: Annotated[
        Path | None, typer.Argument(help="Where to write it (default: the per-user config).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Write a starter config with every option and its default."""
    target = path or user_config_path()
    if target.exists() and not force:
        raise _fail(UsageError(f"{target} already exists (use --force to replace it)"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(defaults_text(), "utf-8")
    console.print(f"Wrote {escape(str(target))}")


def _load_or_exit(config: Path | None) -> LoadedConfig:
    try:
        return load_config(config)
    except VbrError as exc:
        raise _fail(exc) from exc


@config_app.command("show")
def config_show(config: ConfigOpt = None) -> None:
    """Print the effective configuration (defaults + your file), secrets redacted."""
    loaded = _load_or_exit(config)
    source = loaded.source or "built-in defaults only"
    typer.echo(f"# effective configuration; source: {source}\n")
    typer.echo(to_toml(redact(loaded.config.model_dump(mode="json"))), nl=False)


@config_app.command("check")
def config_check(config: ConfigOpt = None) -> None:
    """Validate the configuration and compile the word list."""
    loaded = _load_or_exit(config)
    try:
        lexicon = compile_lexicon(loaded.config.lexicon, base_dir=loaded.base_dir)
    except VbrError as exc:
        raise _fail(exc) from exc
    console.print(f"config: {escape(str(loaded.source or 'built-in defaults only'))}")
    counts: dict[str, int] = {}
    for term in lexicon.terms:
        counts[term.category] = counts.get(term.category, 0) + 1
    listed = ", ".join(f"{name} ({counts.get(name, 0)} terms)" for name in lexicon.categories)
    console.print(f"enabled categories: {escape(listed) or 'none'}")
    for warning in lexicon.warnings:
        console.print(f"[yellow]warning:[/] {escape(warning)}")
    if not lexicon.terms and not lexicon.masked_patterns:
        raise _fail(UsageError("the word list is empty: enable a category or add terms"))
    console.print("[green]✔[/] configuration is valid")


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check FFmpeg, the mute filter, Whisper and credentials."""
    from video_beep_remover.asr.faster_whisper import cuda_available, resolve_model
    from video_beep_remover.media.ffmpeg import MIN_VERSION, FFmpeg
    from video_beep_remover.media.selftest import mute_self_test

    table = Table(show_header=True, header_style="bold")
    for column in ("check", "result", "details"):
        table.add_column(column)
    failed = False

    def row(check: str, ok: bool | None, details: str) -> None:
        nonlocal failed
        failed = failed or ok is False
        mark = {True: "[green]ok[/]", False: "[red]FAIL[/]", None: "[yellow]note[/]"}[ok]
        table.add_row(check, mark, escape(details))

    row("vbr", True, __version__)
    try:
        loaded = load_config(config)
        row("config", True, str(loaded.source or "built-in defaults only"))
        cfg = loaded.config
    except VbrError as exc:
        row("config", False, str(exc))
        console.print(table)
        raise typer.Exit(EXIT_USAGE) from exc

    try:
        ff = FFmpeg(cfg.tools.ffmpeg, cfg.tools.ffprobe)
        version = ff.version
        recent = version.at_least(*MIN_VERSION)
        row(
            "ffmpeg",
            recent,
            f"{version.number or 'unknown version'} ({ff.ffmpeg})"
            + ("" if recent else f"; {MIN_VERSION[0]}.{MIN_VERSION[1]} or later required"),
        )
        missing = [name for name in ("aac", "flac") if name not in ff.encoders]
        row("encoders", not missing, "missing: " + ", ".join(missing) if missing else "aac, flac available")
        problem = mute_self_test(ff)
        row("mute self-test", problem is None, problem or "a synthetic tone was muted and verified")
    except VbrError as exc:
        row("ffmpeg", False, str(exc))

    try:
        import faster_whisper

        row("faster-whisper", True, str(faster_whisper.__version__))
        row("GPU (CUDA)", None, "available" if cuda_available() else "not available; Whisper runs on CPU")
        choice = resolve_model(cfg.transcription, strategy="full", language=cfg.analysis.language)
        try:
            from faster_whisper.utils import download_model

            download_model(choice.name, local_files_only=True)
            row("Whisper model", True, f"{choice.describe()} is downloaded")
        except Exception:
            ok = False if cfg.offline else None
            row("Whisper model", ok, f"{choice.describe()} is not downloaded yet; the first run downloads it")
    except ImportError:
        row("faster-whisper", False, "not installed: pip install faster-whisper")

    key = cfg.subtitles.opensubtitles.api_key
    row("OpenSubtitles key", None, "set" if key else "not set; online subtitle search will be skipped")
    console.print(table)
    raise typer.Exit(EXIT_DEPENDENCY if failed else EXIT_OK)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"vbr {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=version_callback, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Mute a configurable list of words in video files."""


def main() -> None:
    app()
