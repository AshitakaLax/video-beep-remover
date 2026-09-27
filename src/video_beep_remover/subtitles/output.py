"""Subtitles in the cleaned output (DESIGN.md §6.11): the file's own text subtitle streams, masked and
muxed back in place, and a masked copy of the subtitle file the analysis used, next to the output."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.errors import SubtitleError
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import MediaInfo
from video_beep_remover.media.render import StreamPlan
from video_beep_remover.subtitles.acquire import extract_subtitle_streams, extracted_path
from video_beep_remover.subtitles.censor import Mask, censor_subtitles, censored_copy_path
from video_beep_remover.subtitles.parse import read_subtitle_file
from video_beep_remover.ui import UI


@dataclass
class SubtitleStreams:
    plan: StreamPlan  # without the streams that could not be censored
    files: dict[int, Path] = field(default_factory=dict)  # input stream index -> censored file
    report: list[dict[str, Any]] = field(default_factory=list)


def censor_streams(
    ff: FFmpeg,
    info: MediaInfo,
    plan: StreamPlan,
    workdir: Path,
    lexicon: Lexicon,
    mask: Mask,
    ui: UI,
) -> SubtitleStreams:
    """Mask listed words in every subtitle stream the plan censors. Streams the analysis already
    extracted are reused; the others are extracted together, in one pass over the file. A stream that
    cannot be extracted or read is dropped: copying it would keep its words."""
    wanted = [a.stream for a in plan.actions if a.stream.kind == "subtitle" and a.action == "censor"]
    result = SubtitleStreams(plan)
    if not wanted:
        return result
    pairs = [(s.index, s.codec) for s in wanted]
    try:
        if all(extracted_path(workdir, index, codec).is_file() for index, codec in pairs):
            extracted = extract_subtitle_streams(ff, info.path, pairs, workdir)
        else:
            with ui.progress("Reading subtitles", info.duration) as update:
                extracted = extract_subtitle_streams(ff, info.path, pairs, workdir, on_progress=update)
    except SubtitleError as exc:
        for stream in wanted:
            result.plan = result.plan.drop(
                stream.index, f"dropped subtitle stream {stream.describe()}: {exc}"
            )
        return result
    for stream in wanted:
        source = extracted[stream.index]
        try:
            done = censor_subtitles(read_subtitle_file(source), lexicon, mask, fps=info.frame_rate)
        except SubtitleError as exc:
            result.plan = result.plan.drop(
                stream.index, f"dropped subtitle stream {stream.describe()}: {exc}"
            )
            continue
        target = workdir / f"censored-{stream.index}{source.suffix}"
        target.write_text(done.text, "utf-8")
        result.files[stream.index] = target
        result.report.append(
            {
                "stream": stream.index,
                "codec": stream.codec,
                "language": stream.language,
                "masked": done.masked,
            }
        )
    return result


@dataclass(frozen=True)
class CensoredCopy:
    path: Path
    masked: int


def write_censored_copy(
    subtitle: Path,
    *,
    video: Path,
    output: Path,
    language: str | None,
    lexicon: Lexicon,
    mask: Mask,
    fps: float | None,
    overwrite: bool,
    backup: Path | None = None,
) -> CensoredCopy:
    """A masked copy of `subtitle`, next to the output and named after it, so players load it with the
    cleaned file.

    When the cleaned file took the video's place (--in-place, --backup), the copy takes the subtitle
    file's place too. With --backup, `backup` is where the video's original went, and the subtitle file
    is first kept next to it under the matching name ("Movie.en.srt" -> "Movie.orig.en.srt")."""
    target = censored_copy_path(subtitle, video, output, language)
    in_place = target.resolve() == subtitle.resolve()
    if in_place and output.resolve() != video.resolve():
        raise SubtitleError(f"the censored copy would overwrite {subtitle}")
    if target.exists() and not (overwrite or in_place):
        raise SubtitleError(f"{target.name} already exists (use --overwrite)")
    done = censor_subtitles(read_subtitle_file(subtitle, language), lexicon, mask, fps=fps)
    if in_place and backup is not None:
        kept = censored_copy_path(subtitle, video, backup, language)
        if kept.exists():
            raise SubtitleError(f"{kept.name} already exists; {subtitle.name} was left as it is")
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(subtitle, kept)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(done.text, "utf-8")
    tmp.replace(target)
    return CensoredCopy(target, done.masked)
