"""The commands that process videos: clean, scan and render. Each takes video files or folders of them
(DESIGN.md §3.2); batch.py runs the files in turn."""

from pathlib import Path
from typing import Annotated, Any

import typer

from video_beep_remover.batch import Input, Outcome, collect_inputs, run_batch, skip_outputs
from video_beep_remover.cli.app import app
from video_beep_remover.cli.console import ConsoleUI, fail, print_result, setup_logging
from video_beep_remover.cli.options import (
    AudioStreamOpt,
    BackupOpt,
    CategoriesOpt,
    ConfigOpt,
    ContextOpt,
    DeviceOpt,
    EdlOpt,
    InPlaceOpt,
    InputsArg,
    KeepTempOpt,
    LanguageOpt,
    ModelOpt,
    NoFallbackOpt,
    OfflineOpt,
    OutputOpt,
    OverwriteOpt,
    QuietOpt,
    RecursiveOpt,
    ReplaceOpt,
    ReportOpt,
    ReviewOpt,
    SkipExistingOpt,
    StrategyOpt,
    SubtitlesOpt,
    VerboseOpt,
    overrides,
)
from video_beep_remover.config.loader import LoadedConfig, load_config
from video_beep_remover.errors import EXIT_PARTIAL, DependencyError, UsageError, VbrError
from video_beep_remover.outputs import find_report
from video_beep_remover.pipeline import FileResult, Pipeline, RunOptions


def _inputs(
    paths: list[Path], loaded: LoadedConfig, recursive: bool, options: RunOptions
) -> tuple[list[Input], list[FileResult]]:
    """The files to process, and the results of those skipped up front (outputs and backups found in
    folders)."""
    files, skipped = skip_outputs(collect_inputs(paths, recursive), loaded.config.output, options.output)
    many = len(files) > 1
    if many and options.output is not None and options.output.exists() and not options.output.is_dir():
        raise UsageError("with several inputs, --output must be a directory")
    if many and options.report is not None and options.report.is_file():
        raise UsageError("with several inputs, --report must be a directory")
    if many and options.subtitles is not None:
        raise UsageError("--subtitles works with a single input")
    return files, skipped


def _run(
    inputs: list[Path],
    config: Path | None,
    options: RunOptions,
    categories: str | None,
    quiet: bool,
    verbose: bool,
    **flags: Any,
) -> None:
    """clean and scan: analyse each file, and render it unless it is a dry run."""
    setup_logging(verbose)
    ui = ConsoleUI(quiet=quiet)
    try:
        loaded = load_config(config, overrides=overrides(**flags))
        files, skipped = _inputs(inputs, loaded, flags.get("recursive", False), options)
        many = len(files) > 1
        pipeline = Pipeline(loaded, ui=ui, categories=categories.split(",") if categories else None)
    except VbrError as exc:
        raise fail(exc) from exc

    for result in skipped:
        print_result(ui, result)
    failures = 0

    def report(outcome: Outcome) -> None:
        nonlocal failures
        for kind, message in outcome.messages:
            (ui.warn if kind == "warn" else ui.info)(f"{outcome.path.name}: {message}")
        if outcome.error is not None:
            if not many:
                raise fail(outcome.error)
            failures += 1
            ui.error(f"{outcome.path.name}: {outcome.error}")
        elif outcome.result is not None:
            print_result(ui, outcome.result)

    try:
        run_batch(pipeline, files, options, report)
    except DependencyError as exc:
        raise fail(exc) from exc  # would fail for every file: the batch stops
    if failures:
        ui.error(f"{failures} of {len(files)} files failed")
        raise typer.Exit(EXIT_PARTIAL)


@app.command()
def clean(
    inputs: InputsArg,
    config: ConfigOpt = None,
    output: OutputOpt = None,
    backup: BackupOpt = False,
    in_place: InPlaceOpt = False,
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
    review_srt: ReviewOpt = False,
    context: ContextOpt = False,
    replace: ReplaceOpt = False,
    overwrite: OverwriteOpt = False,
    skip_existing: SkipExistingOpt = False,
    recursive: RecursiveOpt = False,
    keep_temp: KeepTempOpt = False,
    verbose: VerboseOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Find the listed words and write a copy of each video with them muted.

    The copy goes next to the original as <name>.clean.<ext>, or takes the original's place with
    --backup or --in-place."""
    options = RunOptions(
        dry_run=dry_run,
        output=output,
        report=report,
        edl=edl,
        review_srt=review_srt,
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
        context=context,
        replace=replace,
        backup=backup,
        in_place=in_place,
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
    review_srt: ReviewOpt = False,
    context: ContextOpt = False,
    replace: ReplaceOpt = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace an existing EDL or review subtitles.")
    ] = False,
    recursive: RecursiveOpt = False,
    verbose: VerboseOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Find the listed words and write a report (and optional EDL), without writing video."""
    options = RunOptions(
        dry_run=True,
        report=report,
        edl=edl,
        review_srt=review_srt,
        overwrite=overwrite,
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
        context=context,
        replace=replace,
    )


@app.command("render")
def render_command(
    inputs: InputsArg,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="The vbr report to render: a file, or a folder for several videos. By default, the "
            "one vbr scan or vbr clean wrote next to each video.",
            show_default=False,
        ),
    ] = None,
    config: ConfigOpt = None,
    output: OutputOpt = None,
    backup: BackupOpt = False,
    in_place: InPlaceOpt = False,
    categories: CategoriesOpt = None,
    audio_stream: AudioStreamOpt = None,
    edl: EdlOpt = False,
    review_srt: ReviewOpt = False,
    overwrite: OverwriteOpt = False,
    skip_existing: SkipExistingOpt = False,
    force: Annotated[
        bool, typer.Option("--force", help="Render even if the report was made for a different file.")
    ] = False,
    recursive: RecursiveOpt = False,
    keep_temp: KeepTempOpt = False,
    verbose: VerboseOpt = False,
    quiet: QuietOpt = False,
) -> None:
    """Mute the spans listed in a report, e.g. one you edited by hand, without detecting anything.

    Only the report's "intervals" are read; the word list is used for the subtitles."""
    setup_logging(verbose)
    ui = ConsoleUI(quiet=quiet)
    options = RunOptions(
        output=output,
        report=report,
        edl=edl,
        review_srt=review_srt,
        overwrite=overwrite,
        skip_existing=skip_existing,
        keep_temp=keep_temp,
    )
    try:
        flags = overrides(audio_stream=audio_stream, backup=backup, in_place=in_place)
        loaded = load_config(config, overrides=flags)
        files, skipped = _inputs(inputs, loaded, recursive, options)
        many = len(files) > 1
        pipeline = Pipeline(loaded, ui=ui, categories=categories.split(",") if categories else None)
    except VbrError as exc:
        raise fail(exc) from exc
    for result in skipped:
        print_result(ui, result)
    failures = 0
    for item in files:
        found = find_report(item.path, report, loaded.config.output, many=many)
        try:
            if found is None:
                if item.from_folder:
                    print_result(
                        ui, FileResult(item.path, "skipped", notes=["no report (vbr scan writes one)"])
                    )
                    continue
                raise UsageError(f"no report for {item.path.name}: run vbr scan first, or pass --report")
            result = pipeline.render_report(
                item.path, found, options, force=force, many=many, from_folder=item.from_folder
            )
        except DependencyError as exc:
            raise fail(exc) from exc  # would fail for every file
        except VbrError as exc:
            if not many:
                raise fail(exc) from exc
            failures += 1
            ui.error(f"{item.path.name}: {exc}")
            continue
        print_result(ui, result)
    if failures:
        ui.error(f"{failures} of {len(files)} files failed")
        raise typer.Exit(EXIT_PARTIAL)
