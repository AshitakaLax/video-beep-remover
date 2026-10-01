"""vbr doctor: checks FFmpeg, the mute filter, Whisper, the optional extras and the credentials. Each
optional part has a function that returns its row: (check, ok, details), ok being None for a note."""

import typer
from rich.markup import escape
from rich.table import Table

from video_beep_remover import __version__
from video_beep_remover.cli.app import app
from video_beep_remover.cli.console import console
from video_beep_remover.cli.options import ConfigOpt
from video_beep_remover.config.loader import load_config
from video_beep_remover.config.schema import Config
from video_beep_remover.errors import EXIT_DEPENDENCY, EXIT_OK, EXIT_USAGE, VbrError


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
    row(*whisperx_status(cfg))
    row(*context_status(cfg))
    row(*voice_status(cfg))

    row(*opensubtitles_status(cfg))
    from video_beep_remover.subtitles.ffsubsync import ffsubsync_command

    command = ffsubsync_command()
    row(
        "ffsubsync",
        None,
        f"installed ({command[0]})"
        if command
        else "not installed; optional: pip install 'video-beep-remover[sync]'",
    )
    console.print(table)
    raise typer.Exit(EXIT_DEPENDENCY if failed else EXIT_OK)


def whisperx_status(cfg: Config) -> tuple[str, bool | None, str]:
    """doctor's WhisperX row: needed only by the whisperx backend, which also needs its models."""
    from importlib.metadata import PackageNotFoundError, version

    name = "WhisperX"
    try:
        installed = version("whisperx")
    except PackageNotFoundError:
        if cfg.transcription.backend != "whisperx":
            return name, None, "not installed; optional: pip install 'video-beep-remover[align]'"
        return (
            name,
            False,
            "not installed, but backend = \"whisperx\": pip install 'video-beep-remover[align]'",
        )
    if cfg.transcription.backend != "whisperx":
        return name, None, f'{installed}, not used (transcription.backend = "faster-whisper")'
    from video_beep_remover.asr.whisperx import status

    try:
        downloaded, details = status(cfg.analysis.language, cfg.transcription.align_model)
    except VbrError as exc:
        return name, False, f"{installed}: {exc}"
    return name, True if downloaded else (False if cfg.offline else None), f"{installed}, aligner {details}"


def context_status(cfg: Config) -> tuple[str, bool | None, str]:
    """doctor's row for context analysis (DESIGN.md §17): its extra, its models and the judge."""
    from importlib.util import find_spec

    from video_beep_remover.asr.faster_whisper import cuda_available
    from video_beep_remover.context import gpu_memory_gb, judge_model, torch_device
    from video_beep_remover.context.models import JUDGE_GPU_GB

    name = "context analysis"
    installed = find_spec("torch") is not None and find_spec("transformers") is not None
    enabled = cfg.context.enabled
    if not installed:
        if not enabled:
            return name, None, "off; optional: pip install 'video-beep-remover[context]', then --context"
        return name, False, "context.enabled, but not installed: pip install 'video-beep-remover[context]'"
    device = torch_device(cfg.transcription.device)
    gpu = gpu_memory_gb() if cfg.context.judge == "auto" and device == "cuda" else None
    judge = judge_model(cfg.context.judge, device, cfg.context.api, offline=cfg.offline, gpu_gb=gpu)
    online = cfg.context.judge == "api"
    models = [cfg.context.classifier] + ([judge] if judge and not online else [])
    try:
        from huggingface_hub import try_to_load_from_cache

        missing = [m for m in models if not isinstance(try_to_load_from_cache(m, "config.json"), str)]
    except ImportError:
        missing = []
    details = f"{'on' if enabled else 'off (--context turns it on)'}; classifier {cfg.context.classifier}; "
    if judge:
        details += f"judge {judge}" + (" (online: the lines it judges are sent to it)" if online else "")
    else:
        why = {"auto": " (no GPU)", "api": " (offline)"}.get(cfg.context.judge, "")
        if gpu is not None:  # too small for the default judge
            why = f" (the GPU has {gpu:.0f} GB; the default judge needs about {JUDGE_GPU_GB:.0f})"
        details += f"no judge{why}"
    if cfg.transcription.device == "auto" and device == "cpu" and cuda_available():
        details += "; PyTorch cannot use the GPU that Whisper uses (is it a CPU-only build?)"
    actions = []
    if cfg.context.harmless == "keep":
        actions.append("keeps harmless uses")
    if cfg.context.sexual == "mute":
        actions.append("mutes sexual lines")
    details += f"; {' and '.join(actions)} (experimental)" if actions else "; report only"
    if missing:
        details += f"; not downloaded yet: {', '.join(missing)} (the first run downloads them)"
    return name, (False if missing and cfg.offline and enabled else True if enabled else None), details


def voice_status(cfg: Config) -> tuple[str, bool | None, str]:
    """doctor's row for voice replacement (DESIGN.md §16): its extra and its models."""
    from importlib.util import find_spec

    name = "voice replacement"
    missing = [m for m in ("torch", "torchaudio", "f5_tts", "demucs", "speechbrain") if find_spec(m) is None]
    enabled = cfg.replace.enabled
    if missing:
        if not enabled:
            return name, None, "off; optional: pip install 'video-beep-remover[voice]', then --replace"
        return (
            name,
            False,
            f"replace.enabled, but {', '.join(missing)} missing: pip install 'video-beep-remover[voice]'",
        )
    details = f"{'on' if enabled else 'off (--replace turns it on)'}; voice model {cfg.replace.model}"
    details += f", separation {cfg.replace.separation}; the voice model's weights are non-commercial"
    return name, True if enabled else None, details


def opensubtitles_status(cfg: Config) -> tuple[str, bool | None, str]:
    """doctor's OpenSubtitles row: is the key set and accepted, and do the credentials log in?"""
    from video_beep_remover.subtitles.opensubtitles import (
        KeyRejected,
        OpenSubtitlesClient,
        OpenSubtitlesError,
    )

    settings = cfg.subtitles.opensubtitles
    name = "OpenSubtitles"
    if not settings.enabled or "opensubtitles" not in cfg.subtitles.sources:
        return name, None, "disabled in the config"
    if not settings.api_key:
        return name, None, "no API key: online subtitle search is skipped (set OPENSUBTITLES_API_KEY)"
    if cfg.offline:
        return name, None, "API key set; not checked (offline)"
    client = OpenSubtitlesClient(
        settings.api_key,
        user_agent=settings.user_agent,
        username=settings.username,
        password=settings.password,
    )
    try:
        client.check_key()
        details = "API key accepted"
        if settings.username and settings.password:
            try:
                client.login()
                allowed = (client.user or {}).get("allowed_downloads")
                details += f"; logged in as {settings.username}" + (
                    f" ({allowed} downloads a day)" if allowed else ""
                )
            except OpenSubtitlesError as exc:
                return name, False, f"API key accepted, but {exc}"
        return name, True, details
    except KeyRejected as exc:
        return name, False, str(exc)
    except OpenSubtitlesError as exc:
        return name, None, f"could not check the API key: {exc}"
    finally:
        client.close()
