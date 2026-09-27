"""Options the commands share, and how flags become configuration overrides (DESIGN.md §3.3, §4.1).

Help texts are Rich markup: a literal "[" is written "\\[", or Rich takes "[voice]" for a style tag."""

from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from video_beep_remover.errors import UsageError


class StrategyChoice(StrEnum):
    hybrid = "hybrid"
    targeted = "targeted"
    full = "full"


class DeviceChoice(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


def overrides(**flags: Any) -> dict[str, Any]:
    """The configuration keys the command-line flags set (flags left at None or False set nothing)."""
    keys = {
        "strategy": "analysis.strategy",
        "language": "analysis.language",
        "audio_stream": "analysis.audio_stream",
        "model": "transcription.model",
        "device": "transcription.device",
    }
    found = {
        keys[name]: value.value if isinstance(value, Enum) else value
        for name, value in flags.items()
        if name in keys and value is not None
    }
    if flags.get("no_fallback"):
        found["analysis.fallback_to_full"] = False
    if flags.get("offline"):
        found["offline"] = True
    if flags.get("context"):
        found["context.enabled"] = True
    if flags.get("replace"):
        found["replace.enabled"] = True
    if flags.get("backup") and flags.get("in_place"):
        raise UsageError("--backup and --in-place cannot be combined")
    if flags.get("backup"):
        found["output.mode"] = "backup"
    if flags.get("in_place"):
        found["output.mode"] = "in_place"
    return found


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
ReportOpt = Annotated[
    Path | None,
    typer.Option(help="Where to write the JSON report: a file, or a folder for several videos."),
]
EdlOpt = Annotated[bool, typer.Option("--edl", help="Also write <input>.edl, a mute list for Kodi/MPlayer.")]
ReviewOpt = Annotated[
    bool,
    typer.Option("--review-srt", help="Also write .review.srt: one cue per muted span, for spot checks."),
]
ContextOpt = Annotated[
    bool,
    typer.Option(
        "--context",
        help="Also read each listed word and the script in context with local models, and add the "
        "verdicts to the report (needs the \\[context] extra; acting on them is set in \\[context]).",
    ),
]
ReplaceOpt = Annotated[
    bool,
    typer.Option(
        "--replace",
        help="Say a milder word in the speaker's voice instead of muting, where one fits (experimental; "
        "needs the \\[voice] extra, and a GPU in practice).",
    ),
]
OutputOpt = Annotated[
    Path | None, typer.Option("--output", "-o", help="Output file, or a folder for several videos.")
]
BackupOpt = Annotated[
    bool,
    typer.Option(
        "--backup",
        help="Put the cleaned file in the original's place, and keep the unmodified original as "
        "<name>.orig.<ext> (output.backup_path).",
    ),
]
InPlaceOpt = Annotated[
    bool,
    typer.Option("--in-place", help="Put the cleaned file in the original's place, with no backup."),
]
OverwriteOpt = Annotated[bool, typer.Option("--overwrite", help="Replace existing outputs.")]
SkipExistingOpt = Annotated[bool, typer.Option("--skip-existing", help="Skip inputs whose output exists.")]
KeepTempOpt = Annotated[bool, typer.Option("--keep-temp", help="Keep temporary files for debugging.")]
RecursiveOpt = Annotated[bool, typer.Option("--recursive", "-r", help="Search folders recursively.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show debug output.")]
QuietOpt = Annotated[bool, typer.Option("--quiet", "-q", help="Only show results and errors.")]
