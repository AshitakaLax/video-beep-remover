"""vbr cache: inspect or clear the cache of downloaded subtitles, transcripts and judge answers
(DESIGN.md §8.3)."""

from typing import Annotated

import typer
from rich.markup import escape

from video_beep_remover.cli.app import cache_app
from video_beep_remover.cli.console import console, load_or_exit
from video_beep_remover.cli.options import ConfigOpt
from video_beep_remover.config.loader import cache_root


@cache_app.command("info")
def cache_info(config: ConfigOpt = None) -> None:
    """Show where the cache is and what it holds."""
    from video_beep_remover.asr.cache import TranscriptCache
    from video_beep_remover.subtitles.cache import SubtitleCache

    cfg = load_or_exit(config).config
    root = cache_root(cfg)
    subtitles, subtitle_bytes = SubtitleCache(root).usage()
    transcripts, transcript_bytes = TranscriptCache(root).usage()
    console.print(f"cache: {escape(str(root))}")
    console.print(f"downloaded subtitles: {subtitles} files, {subtitle_bytes / 1024:.0f} KiB")
    state = "" if cfg.cache.transcripts else " (not kept: cache.transcripts = false)"
    console.print(
        f"transcripts: {transcripts} files, {transcript_bytes / 1024**2:.1f} MiB "
        f"of at most {cfg.cache.max_size_gb:g} GB{state}"
    )
    answers = [path for path in (root / "context").glob("*.jsonl") if path.is_file()]
    if answers:
        size = sum(path.stat().st_size for path in answers)
        console.print(f"context judge answers: {len(answers)} files, {size / 1024:.0f} KiB")


@cache_app.command("clear")
def cache_clear(
    config: ConfigOpt = None,
    subtitles: Annotated[
        bool, typer.Option("--subtitles", help="Only the downloaded subtitles (downloading costs quota).")
    ] = False,
    transcripts: Annotated[bool, typer.Option("--transcripts", help="Only the transcripts.")] = False,
) -> None:
    """Delete what the cache holds: everything (context judge answers too), or only the subtitles or
    the transcripts."""
    from video_beep_remover.asr.cache import TranscriptCache
    from video_beep_remover.subtitles.cache import SubtitleCache

    root = cache_root(load_or_exit(config).config)
    both = not subtitles and not transcripts
    if subtitles or both:
        console.print(f"removed {SubtitleCache(root).clear()} downloaded subtitle files")
    if transcripts or both:
        console.print(f"removed {TranscriptCache(root).clear()} transcript files")
    if both:
        answers = [path for path in (root / "context").glob("*.jsonl") if path.is_file()]
        for path in answers:
            path.unlink(missing_ok=True)
        if answers:
            console.print(f"removed {len(answers)} files of context judge answers")
