"""Write the cleaned file: mute spans with afade driven by a command file (DESIGN.md §6.11)."""

import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from video_beep_remover.config.schema import OutputConfig
from video_beep_remover.errors import DependencyError, RenderError, VbrError
from video_beep_remover.languages import lang_matches
from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import MediaInfo, StreamInfo, parse_probe
from video_beep_remover.models import CensorInterval

log = logging.getLogger(__name__)

FRAMES_PER_SECOND = 100  # asetnsamples caps frames at 10 ms so every command range contains frame starts
MIN_GAP_S = 0.05  # closer spans are merged: the range between them must contain frame starts
MIN_SPAN_S = 0.06  # shorter spans are lengthened, for the same reason
VERIFY_MARGIN_S = 0.025  # distance kept from the fades when checking that a span is silent
SILENCE_DBFS = -60.0
MP4_LIKE = frozenset({".mp4", ".m4v", ".m4a", ".mov"})
CENSORED_TAG = "VBR_CENSORED"  # "<version>;<config hash>" on every output, so later runs can skip it

# Encoder for the same codec as the source, when FFmpeg has it.
_SAME_CODEC = {
    "aac": "aac", "ac3": "ac3", "eac3": "eac3", "mp3": "libmp3lame", "opus": "libopus",
    "vorbis": "libvorbis", "flac": "flac", "alac": "alac",
}  # fmt: skip
# Otherwise (DTS, TrueHD, missing encoder): what suits the output container.
_CONTAINER_FALLBACK = {
    ".mkv": "flac", ".mka": "flac", ".webm": "libopus", ".mp4": "aac", ".m4v": "aac", ".m4a": "aac",
    ".mov": "aac", ".ts": "ac3", ".m2ts": "ac3", ".mts": "ac3", ".avi": "ac3", ".mpg": "ac3",
    ".mpeg": "ac3", ".vob": "ac3",
}  # fmt: skip
_DISPOSITIONS = (
    "default", "dub", "original", "comment", "lyrics", "karaoke", "forced", "hearing_impaired",
    "visual_impaired", "clean_effects",
)  # fmt: skip


@dataclass(frozen=True)
class StreamAction:
    stream: StreamInfo
    action: Literal["copy", "censor"]


@dataclass(frozen=True)
class StreamPlan:
    actions: tuple[StreamAction, ...]  # output order; dropped streams are not listed
    notes: tuple[str, ...]

    @property
    def censored(self) -> list[StreamInfo]:
        return [a.stream for a in self.actions if a.action == "censor"]

    def drop(self, index: int, note: str) -> "StreamPlan":
        """The same plan without input stream `index`, and a note saying why."""
        return StreamPlan(tuple(a for a in self.actions if a.stream.index != index), (*self.notes, note))


def plan_streams(
    info: MediaInfo, analysed: StreamInfo, output: OutputConfig, language: str, lexicon_language: str = "en"
) -> StreamPlan:
    """Decide what happens to every input stream (output.other_audio_streams, output.subtitle_streams)."""
    actions: list[StreamAction] = []
    notes: list[str] = []
    for stream in info.streams:
        if stream.kind == "audio":
            if stream.index == analysed.index:
                actions.append(StreamAction(stream, "censor"))
                continue
            policy = output.other_audio_streams
            if policy == "auto":
                same_dialogue = (
                    not stream.is_commentary
                    and not stream.is_audio_description
                    and (lang_matches(stream.language, language) or stream.language == analysed.language)
                )
                policy = "censor" if same_dialogue else "drop"
            if policy == "censor":
                actions.append(StreamAction(stream, "censor"))
                notes.append(f"audio stream {stream.describe()} gets the same mutes as the analysed stream")
            elif policy == "copy":
                actions.append(StreamAction(stream, "copy"))
                notes.append(f"audio stream {stream.describe()} copied unchanged: it was not analysed")
            else:
                notes.append(
                    f"dropped audio stream {stream.describe()}: it was not analysed, "
                    "so it could still contain listed words"
                )
        elif stream.kind == "subtitle":
            if output.subtitle_streams == "copy":
                actions.append(StreamAction(stream, "copy"))
            elif output.subtitle_streams == "drop":
                notes.append(
                    f'dropped subtitle stream {stream.describe()} (output.subtitle_streams = "drop")'
                )
            elif not stream.is_text_subtitle:
                actions.append(StreamAction(stream, "copy"))
                notes.append(
                    f"subtitle stream {stream.describe()} copied uncensored: "
                    "image-based subtitles cannot be edited"
                )
            else:
                actions.append(StreamAction(stream, "censor"))
                if stream.language and not lang_matches(stream.language, lexicon_language):
                    notes.append(
                        f"subtitle stream {stream.describe()} is masked with the {lexicon_language!r} "
                        "word list, which does not cover its language"
                    )
        elif stream.kind in ("video", "attachment"):
            actions.append(StreamAction(stream, "copy"))
        else:
            notes.append(f"dropped {stream.kind} stream #{stream.index} ({stream.codec or 'unknown codec'})")
    return StreamPlan(tuple(actions), tuple(notes))


def normalize_intervals(intervals: Sequence[CensorInterval], duration: float) -> list[CensorInterval]:
    """Sort, clamp, lengthen very short spans and merge very close ones (see MIN_GAP_S, MIN_SPAN_S)."""
    spans: list[list[float]] = []
    for interval in sorted(intervals, key=lambda i: i.start):
        start, end = max(0.0, interval.start), min(duration, interval.end)
        if end - start < MIN_SPAN_S:
            middle = (start + end) / 2
            start, end = max(0.0, middle - MIN_SPAN_S / 2), min(duration, middle + MIN_SPAN_S / 2)
        if spans and start - spans[-1][1] < MIN_GAP_S:
            spans[-1][1] = max(spans[-1][1], end)
        elif end > start:
            spans.append([start, end])
    return [CensorInterval(start, end) for start, end in spans]


def fade_for(interval: CensorInterval, fade: float) -> float:
    return min(fade, interval.duration / 3)


def command_file(intervals: Sequence[CensorInterval], label: str, fade: float) -> tuple[str, str]:
    """asendcmd commands that re-arm afade@label around each interval, and afade's initial options.

    Each line fires once, when the first frame starting inside its time range arrives. The ranges are
    stretches where old and new settings give the same gain (full volume between spans, silence inside).
    """
    lines: list[str] = []
    initial = "t=in:ss=0:ns=1"  # full volume from the first sample
    previous_end = 0.0
    for number, interval in enumerate(intervals):
        f = fade_for(interval, fade)
        start, end = interval.start, interval.end
        if number == 0 and start <= 0:
            initial = f"t=out:ss=0:d={f:.6f}"
        else:
            lines.append(
                f"{previous_end:.6f}-{start:.6f} [enter] afade@{label} t out, "
                f"[enter] afade@{label} st {start:.6f}, [enter] afade@{label} d {f:.6f};"
            )
        lines.append(
            f"{start + f:.6f}-{end - f:.6f} [enter] afade@{label} t in, "
            f"[enter] afade@{label} st {end - f:.6f}, [enter] afade@{label} d {f:.6f};"
        )
        previous_end = end
    return "\n".join(lines) + "\n", f"{initial}:curve=qsin"


def choose_encoder(
    stream: StreamInfo, suffix: str, encoders: frozenset[str], codec: str, bitrate: str
) -> tuple[str, str | None]:
    """Encoder and bitrate (None for lossless) for a censored stream (output.audio_codec/_bitrate)."""
    if codec != "auto":
        if codec not in encoders:
            raise DependencyError(f"FFmpeg has no {codec!r} encoder (output.audio_codec)")
        encoder = codec
    else:
        source = stream.codec or ""
        encoder = _SAME_CODEC.get(source) or (source if source.startswith("pcm_") else "")
        if encoder not in encoders:
            encoder = _CONTAINER_FALLBACK.get(suffix, "aac")
        if encoder not in encoders:
            options = ("libvorbis",) if suffix == ".webm" else ("aac",)
            encoder = next((name for name in options if name in encoders), "")
        if not encoder:
            raise DependencyError(f"FFmpeg has no audio encoder usable for {suffix} output")
    if encoder in ("flac", "alac") or encoder.startswith("pcm_"):
        return encoder, None
    if bitrate != "auto":
        return encoder, bitrate
    return encoder, _default_bitrate(encoder, stream)


def _default_bitrate(encoder: str, stream: StreamInfo) -> str:
    channels = stream.channels or 2
    source = stream.bit_rate if _SAME_CODEC.get(stream.codec or "") == encoder else None
    if encoder in ("ac3", "eac3"):
        default, cap = (640_000 if channels > 2 else 192_000), (640_000 if encoder == "ac3" else 6_144_000)
    elif encoder == "libopus":
        default, cap = max(96_000, 48_000 * channels), 512_000
    elif encoder == "libmp3lame":
        default, cap = 192_000, 320_000
    else:
        default, cap = max(128_000, 64_000 * channels), 1_536_000
    return f"{round(min(source or default, cap) / 1000)}k"


def subtitle_encoder(stream: StreamInfo, suffix: str) -> str:
    """The encoder for a censored text subtitle stream: the same codec where the container allows it."""
    if suffix in MP4_LIKE:
        return "mov_text"
    if suffix == ".webm" or stream.codec == "webvtt":
        return "webvtt"
    return "ass" if stream.codec in ("ass", "ssa") else "srt"


def disposition_value(stream: StreamInfo) -> str:
    flags = [name for name in _DISPOSITIONS if stream.disposition.get(name)]
    return "+".join(flags) if flags else "0"


@dataclass(frozen=True)
class RenderCommand:
    args: list[str]
    files: dict[str, str]  # file name (in the job directory) -> content
    encoders: dict[int, str]  # input stream index -> encoder, for re-encoded streams
    censored_positions: list[int]  # output audio positions (0:a:N) that were muted


def _tags(kind: str, position: int, stream: StreamInfo) -> list[str]:
    """Language, title and disposition of an output stream that does not come straight from its input
    stream (filtered audio, or subtitles read from a censored file)."""
    args: list[str] = []
    if stream.language:
        args += [f"-metadata:s:{kind}:{position}", f"language={stream.language}"]
    if stream.title:
        args += [f"-metadata:s:{kind}:{position}", f"title={stream.title}"]
    return [*args, f"-disposition:{kind}:{position}", disposition_value(stream)]


def build_command(
    ff: FFmpeg,
    info: MediaInfo,
    plan: StreamPlan,
    intervals: Sequence[CensorInterval],
    *,
    fade: float,
    output: OutputConfig,
    target: Path,
    subtitle_files: Mapping[int, Path] | None = None,
    tag: str | None = None,
) -> RenderCommand:
    """The FFmpeg command for the plan. `subtitle_files` holds the censored text of each subtitle
    stream the plan censors (by input stream index); `tag` is written as the VBR_CENSORED tag."""
    suffix = target.suffix.lower()
    subtitle_files = subtitle_files or {}
    files: dict[str, str] = {}
    graph: list[str] = []
    inputs: list[str] = ["-i", file_arg(info.path)]
    maps: list[str] = []
    codec_args: list[str] = []
    meta_args: list[str] = []
    encoders: dict[int, str] = {}
    positions: list[int] = []
    audio_position = subtitle_position = 0
    for action in plan.actions:
        stream = action.stream
        if stream.kind == "subtitle" and action.action == "censor":
            inputs += ["-i", file_arg(subtitle_files[stream.index])]
            maps += ["-map", f"{len(inputs) // 2 - 1}:0"]
            codec_args += [f"-c:s:{subtitle_position}", subtitle_encoder(stream, suffix)]
            meta_args += _tags("s", subtitle_position, stream)
        elif action.action == "censor" and intervals:
            label = f"mute{len(positions)}"
            text, initial = command_file(intervals, label, fade)
            files[f"{label}.cmd"] = text
            frame = max(1, round((stream.sample_rate or 48_000) / FRAMES_PER_SECOND))
            graph.append(
                f"[0:{stream.index}]asetnsamples=n={frame}:p=0,asendcmd=f={label}.cmd,"
                f"afade@{label}={initial}[out{label}]"
            )
            maps += ["-map", f"[out{label}]"]
            encoder, bitrate = choose_encoder(
                stream, suffix, ff.encoders, output.audio_codec, output.audio_bitrate
            )
            codec_args += [f"-c:a:{audio_position}", encoder]
            if bitrate:
                codec_args += [f"-b:a:{audio_position}", bitrate]
            # Filtered streams lose their per-stream tags: re-apply them from the probe.
            meta_args += _tags("a", audio_position, stream)
            encoders[stream.index] = encoder
            positions.append(audio_position)
        else:
            maps += ["-map", f"0:{stream.index}"]
        if stream.kind == "audio":
            audio_position += 1
        elif stream.kind == "subtitle":
            subtitle_position += 1

    args = list(inputs)
    if graph:
        files["graph.txt"] = ";\n".join(graph) + "\n"
        args += ff.filter_script_args("graph.txt")
    args += [*maps, "-c", "copy", *codec_args, *meta_args]
    args += ["-map_metadata", "0", "-map_chapters", "0", "-max_muxing_queue_size", "4096"]
    if tag:
        args += ["-metadata", f"{CENSORED_TAG}={tag}"]
    if suffix in MP4_LIKE:
        # use_metadata_tags: MP4 keeps only its standard tags otherwise, and VBR_CENSORED is not one.
        args += ["-movflags", "+faststart+use_metadata_tags" if tag else "+faststart"]
    args.append(file_arg(target))
    return RenderCommand(args=args, files=files, encoders=encoders, censored_positions=positions)


def level_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20 * math.log10(rms + 1e-12)


def timeline_shift(ff: FFmpeg, info: MediaInfo, plan: StreamPlan, rendered: Path) -> float:
    """How much later the content sits on the rendered file's timeline than on the input's, in seconds.

    Both timelines start at their file's earliest timestamp. Re-encoding can move that: an AAC
    encoder's priming packet sits before the first sample, 21 ms at 48 kHz and 46 ms at 22.05 kHz,
    and in Matroska it can make the new file start earlier than the old one did. FFmpeg shifts every
    stream alike, so a stream-copied one (usually the video) shows the shift. Returns 0 when there is
    no copied stream with a start time."""
    try:
        rendered_info = parse_probe(rendered, ff.probe(rendered))
    except VbrError:
        return 0.0
    for position, action in enumerate(plan.actions):
        stream = action.stream
        if action.action != "copy" or stream.start_time is None or stream.is_attached_picture:
            continue
        if stream.kind not in ("video", "audio", "subtitle") or position >= len(rendered_info.streams):
            continue
        copied = rendered_info.streams[position]
        if copied.start_time is None:
            continue
        return (copied.start_time - rendered_info.start_time) - (stream.start_time - info.start_time)
    return 0.0


def verify_muted(
    ff: FFmpeg,
    path: Path,
    positions: Sequence[int],
    intervals: Sequence[CensorInterval],
    fade: float,
    *,
    shift: float = 0.0,
    workers: int = 4,
) -> tuple[int, list[str]]:
    """Decode the core of every muted span (the span minus its fades) and require silence. `shift` is
    how much later the content sits in `path` than in the input (see timeline_shift).

    FFmpeg ignores a filter command it rejects without an error, so this is what proves the mutes happened.
    Returns (spans checked, failures).
    """
    jobs = []
    for position in positions:
        for interval in intervals:
            f = fade_for(interval, fade)
            start = interval.start + shift + f + VERIFY_MARGIN_S
            length = interval.end - f - VERIFY_MARGIN_S - start
            if length >= 0.02:
                jobs.append((position, interval, start, length))

    def check(job: tuple[int, CensorInterval, float, float]) -> str | None:
        position, interval, start, length = job
        data = ff.capture([
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", file_arg(path),
            "-map", f"0:a:{position}", "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1",
        ])  # fmt: skip
        samples = np.frombuffer(data, dtype=np.float32)
        where = f"audio stream a:{position}, {interval.start:.2f}-{interval.end:.2f} s"
        if samples.size == 0:
            return f"{where}: no audio decoded"
        level = level_dbfs(samples)
        return f"{where}: {level:.1f} dBFS" if level > SILENCE_DBFS else None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        failures = [result for result in pool.map(check, jobs) if result]
    return len(jobs), failures


@dataclass(frozen=True)
class RenderResult:
    output: Path
    encoders: dict[int, str]
    intervals: tuple[CensorInterval, ...]
    verified_spans: int
    timeline_shift: float = 0.0  # the muted spans sit this much later in the output (see timeline_shift)


def render(
    ff: FFmpeg,
    info: MediaInfo,
    plan: StreamPlan,
    intervals: Sequence[CensorInterval],
    *,
    output: Path,
    fade: float,
    output_config: OutputConfig,
    workdir: Path,
    subtitle_files: Mapping[int, Path] | None = None,
    tag: str | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> RenderResult:
    """Render to <name>.partial<ext>, verify every muted span, then rename. Nothing half-written survives."""
    spans = normalize_intervals(intervals, info.duration)
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    command = build_command(
        ff,
        info,
        plan,
        spans,
        fade=fade,
        output=output_config,
        target=partial,
        subtitle_files=subtitle_files,
        tag=tag,
    )
    for name, text in command.files.items():
        (workdir / name).write_text(text, "utf-8")
    try:
        ff.run(command.args, cwd=workdir, on_progress=on_progress)
        shift = timeline_shift(ff, info, plan, partial)
        checked, failures = verify_muted(
            ff, partial, command.censored_positions, spans, fade, shift=shift, workers=os.cpu_count() or 4
        )
        if failures:
            shown = "\n  ".join(failures[:10])
            more = f"\n  … and {len(failures) - 10} more" if len(failures) > 10 else ""
            raise RenderError(
                f"verification failed: {len(failures)} of {checked} muted spans are not silent; "
                f"the output was deleted.\n  {shown}{more}"
            )
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    log.debug("rendered %s, verified %d spans", output, checked)
    return RenderResult(
        output=output,
        encoders=command.encoders,
        intervals=tuple(spans),
        verified_spans=checked,
        timeline_shift=shift,
    )
