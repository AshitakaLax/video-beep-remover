"""vbr subs: the subtitle candidates of each video, their ranking and the sync check (DESIGN.md §6.3,
§6.6)."""

import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from video_beep_remover.batch import collect_inputs
from video_beep_remover.cli.app import app
from video_beep_remover.cli.console import ConsoleUI, console, fail, setup_logging
from video_beep_remover.cli.options import (
    AudioStreamOpt,
    ConfigOpt,
    DeviceOpt,
    InputsArg,
    LanguageOpt,
    OfflineOpt,
    RecursiveOpt,
    SubtitlesOpt,
    VerboseOpt,
    overrides,
)
from video_beep_remover.config.loader import load_config
from video_beep_remover.errors import EXIT_OK, EXIT_PARTIAL, EXIT_PROCESSING, UsageError, VbrError
from video_beep_remover.pipeline import Pipeline


def _show(pipeline: Pipeline, video: Path, subtitles: Path | None, no_sync: bool, save: Path | None) -> bool:
    """List the candidates for one video, check their sync, and save the chosen subtitles. False if
    none is usable."""
    from video_beep_remover import guided
    from video_beep_remover.media.audio import SeekingAudioSource
    from video_beep_remover.media.probe import probe, select_audio_stream
    from video_beep_remover.subtitles.acquire import SubtitleCandidate, SubtitleLoader, SubtitleSearch
    from video_beep_remover.subtitles.save import save_subtitles

    cfg = pipeline.config
    info = probe(pipeline.ff, video)
    online = pipeline.online_source(info) if subtitles is None else None
    search = SubtitleSearch(info, cfg.subtitles, explicit=subtitles, online=online)
    if online is not None:
        asked = "; then ".join(
            ", ".join(f"{key}={value}" for key, value in params.items())
            for params in online.planned_searches()
        )
        verb = "sends" if online.client is not None else "would send"
        console.print(
            f"[dim]OpenSubtitles {verb}: {escape(asked or 'nothing')} "
            f"(languages: {escape(', '.join(cfg.subtitles.languages))})[/]"
        )
    candidates: list[SubtitleCandidate] = []
    with console.status("Looking for subtitles"):
        for source in search.sources:
            found = search.search(source)
            candidates += found.candidates
            for note in found.notes:
                console.print(f"[dim]{escape(note)}[/]")
    if not candidates:
        console.print("no subtitle candidates found")
        return False
    table = Table(show_header=True, header_style="bold")
    for column in ("#", "source", "subtitles", "language", "SDH", "timed for this file", "score"):
        table.add_column(column)
    for number, candidate in enumerate(candidates, 1):
        table.add_row(
            str(number),
            candidate.source + (" (cached)" if candidate.cached else ""),
            escape(candidate.label),
            candidate.language or "?",
            "yes" if candidate.hearing_impaired else "",
            "yes" if candidate.trusted else "no",
            f"{candidate.score:.0f}" if candidate.source != "explicit" else "",
        )
    console.print(table)

    with tempfile.TemporaryDirectory(prefix="vbr-") as tmp:
        workdir = Path(tmp)
        if no_sync:
            top = candidates[0]
            text = SubtitleLoader(pipeline.ff, video, workdir, [top], online=online).text(top)
        else:
            stream = select_audio_stream(info, cfg.analysis.language, cfg.analysis.audio_stream)
            audio = SeekingAudioSource(pipeline.ff, video, stream.index)
            with console.status("Checking the sync (transcribes a few seconds around six cues)"):
                selection = guided.select_subtitles(
                    pipeline.guided_context(), info, audio, stream.index, workdir, search
                )
            for entry in selection.tried:
                used = entry.get("result") == "used"
                sync = entry.get("sync")
                detail = ""
                if sync:
                    fidelity = "unknown" if sync["fidelity"] is None else f"{sync['fidelity']:.2f}"
                    detail = (
                        f" [{sync['matched']}/{sync['anchors']} anchors, offset {sync['offset']:+.2f} s, "
                        f"scale {sync['scale']:.4f}, error {sync['error']:.2f} s, fidelity {fidelity}]"
                    )
                if entry.get("ffsubsync"):
                    detail += f" [ffsubsync: {entry['ffsubsync']}]"
                mark = "[green]✔[/]" if used else "[red]✘[/]"
                console.print(
                    f"{mark} {escape(entry['label'])}: {escape(str(entry['result']))}{escape(detail)}"
                )
            if selection.chosen is None:
                return False
            text = selection.chosen.text
        if save is not None:
            save_subtitles(text, save, fps=info.frame_rate)
            console.print(f"Wrote {escape(str(save))}")
    return True


@app.command("subs")
def subs_command(
    inputs: InputsArg,
    config: ConfigOpt = None,
    subtitles: SubtitlesOpt = None,
    offline: OfflineOpt = False,
    language: LanguageOpt = None,
    audio_stream: AudioStreamOpt = None,
    device: DeviceOpt = None,
    no_sync: Annotated[bool, typer.Option("--no-sync", help="Only list the candidates.")] = False,
    save: Annotated[
        Path | None,
        typer.Option(
            "--save",
            help="Write the chosen subtitles here; the extension sets the format. With several videos, a "
            "folder: each is saved as <video name>.srt.",
        ),
    ] = None,
    recursive: RecursiveOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Show subtitle candidates, their ranking and the sync check. Exits with 1 if none is usable (4 if
    some of several videos have none)."""
    setup_logging(verbose)
    flags = overrides(offline=offline, language=language, audio_stream=audio_stream, device=device)
    try:
        loaded = load_config(config, overrides=flags)
        videos = [item.path for item in collect_inputs(inputs, recursive)]
        many = len(videos) > 1
        if subtitles is not None and not subtitles.is_file():
            raise UsageError(f"subtitle file not found: {subtitles}")
        if many and subtitles is not None:
            raise UsageError("--subtitles works with a single video")
        if many and save is not None and save.suffix and not save.is_dir():
            raise UsageError("with several videos, --save must be a folder")
        pipeline = Pipeline(loaded, ui=ConsoleUI(quiet=True))
    except VbrError as exc:
        raise fail(exc) from exc
    missing = 0
    for video in videos:
        if many:
            console.print(f"\n[bold]{escape(video.name)}[/]")
        target = save
        if save is not None and (many or save.is_dir()):
            target = save / f"{video.stem}.srt"
        try:
            usable = _show(pipeline, video, subtitles, no_sync, target)
        except VbrError as exc:
            if not many:
                raise fail(exc) from exc
            ConsoleUI().error(f"{video.name}: {exc}")
            usable = False
        missing += not usable
    if missing:
        raise typer.Exit(EXIT_PARTIAL if many and missing < len(videos) else EXIT_PROCESSING)
    raise typer.Exit(EXIT_OK)
