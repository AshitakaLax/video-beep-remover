"""Subtitle-guided analysis: the `targeted` and `hybrid` strategies (DESIGN.md §6.3-6.9 and §7).

Subtitles show where listed words are likely to be; only a few seconds of audio around those cues
are transcribed. Anything that makes the subtitles untrustworthy raises Fallback, and the pipeline
then transcribes the whole track (or fails, with `fallback_to_full = false`)."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_beep_remover.asr.base import Clip, Transcriber
from video_beep_remover.asr.faster_whisper import ModelChoice
from video_beep_remover.asr.vad import SpeechDetector, trim_to_speech
from video_beep_remover.config.schema import Config
from video_beep_remover.detect.confirm import (
    attribute,
    dedupe,
    detect_in_windows,
    resolve_unconfirmed,
    split_confirmed,
)
from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.detect.planner import (
    FlaggedCue,
    audio_seconds,
    flag_cues,
    flagged_windows,
    plan_windows,
    search_padding,
    uncovered_speech,
)
from video_beep_remover.errors import SubtitleError
from video_beep_remover.media.audio import Audio, AudioSource
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import MediaInfo
from video_beep_remover.models import Cue, Detection, Window, Word
from video_beep_remover.subtitles import ffsubsync
from video_beep_remover.subtitles.acquire import (
    OnlineSource,
    SubtitleCandidate,
    SubtitleLoader,
    SubtitleSearch,
)
from video_beep_remover.subtitles.parse import parse_subtitles
from video_beep_remover.subtitles.sync import STANDARD_RATIOS, SyncResult, check_sync, snap
from video_beep_remover.ui import UI

MIN_FLAGS_TO_ESCALATE = 3  # below this many strong flags, a poor confirmation rate proves nothing
MIN_CLIP_S = 0.1
READ_WORKERS = 4


class Fallback(Exception):
    """The subtitles cannot guide this run. `report` holds what was learned before giving up."""

    def __init__(self, reason: str, report: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.report = report


@dataclass
class Context:
    """What the guided analysis needs from the pipeline."""

    config: Config
    lexicon: Lexicon
    prompt: str | None
    ui: UI
    ff: FFmpeg
    transcriber: Callable[[str], tuple[ModelChoice, Transcriber]]  # by role: "targeted", "anchor", ...
    detect_speech: SpeechDetector
    online: Callable[[MediaInfo], OnlineSource | None] = lambda info: None  # OpenSubtitles, if usable


@dataclass(frozen=True)
class SubtitleChoice:
    candidate: SubtitleCandidate
    text: str  # the file as loaded, before cleaning
    cues: list[Cue]
    sync: SyncResult


@dataclass
class SubtitleSelection:
    chosen: SubtitleChoice | None
    tried: list[dict[str, Any]] = field(default_factory=list)  # report entries, in the order tried
    notes: tuple[str, ...] = ()


@dataclass
class GuidedResult:
    detections: list[Detection]
    words: int
    model: ModelChoice | None  # None when nothing needed transcribing
    report: dict[str, Any]


def _sync_report(sync: SyncResult) -> dict[str, Any]:
    return {
        "checked": sync.checked,
        "scale": sync.model.scale,
        "offset": round(sync.model.offset, 3),
        "error": round(sync.model.error, 3),
        "anchors": len(sync.anchors),
        "matched": sync.matched,
        "fidelity": None if sync.fidelity is None else round(sync.fidelity, 3),
    }


def _candidate_report(candidate: SubtitleCandidate) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source": candidate.source,
        "label": candidate.label,
        "path": str(candidate.path) if candidate.path else None,
        "stream": candidate.stream,
        "language": candidate.language,
        "hearing_impaired": candidate.hearing_impaired,
        "trusted": candidate.trusted,
    }
    if candidate.source == "opensubtitles":
        report |= {
            "file_id": candidate.file_id,
            "release": candidate.release,
            "fps": candidate.fps,
            "cached": candidate.cached,
        }
    return report


def frame_rate_ratio(candidate: SubtitleCandidate, text: str, video_fps: float | None) -> float:
    """The scale to apply up front when the subtitles were timed for another frame rate, e.g. a
    25 fps PAL release of a 23.976 fps film (DESIGN.md §6.6 step 2). Frame-based files (MicroDVD)
    are already read at the video's frame rate, so they need none."""
    if not candidate.fps or not video_fps:
        return 1.0
    try:
        from pysubs2.formats import autodetect_format

        if autodetect_format(text) == "microdvd":
            return 1.0
    except Exception:  # an undetectable format fails to parse later anyway
        return 1.0
    ratio = snap(candidate.fps / video_fps)
    return ratio if ratio != 1.0 and ratio in STANDARD_RATIOS else 1.0


def describe_sync(sync: SyncResult) -> str:
    if not sync.checked:
        return "not checked (analysis.sync.anchors = 0)"
    fidelity = "unknown" if sync.fidelity is None else f"{sync.fidelity:.2f}"
    return (
        f"{sync.matched}/{len(sync.anchors)} anchors · offset {sync.model.offset:+.2f} s · "
        f"error {sync.model.error:.2f} s · fidelity {fidelity}"
    )


def anchor_transcriber(ctx: Context, audio: AudioSource) -> Callable[[float, float], Sequence[Word]]:
    _, model = ctx.transcriber("anchor")
    language = ctx.config.analysis.language

    def transcribe(start: float, end: float) -> Sequence[Word]:
        clip, _, _ = trim_to_speech(Clip(start, audio.read(start, end)), ctx.detect_speech)
        if clip.duration < MIN_CLIP_S:
            return []
        return model.transcribe([clip], language=language, prompt=None)[0]

    return transcribe


def select_subtitles(
    ctx: Context,
    info: MediaInfo,
    audio: AudioSource,
    stream_index: int,
    workdir: Path,
    search: SubtitleSearch,
) -> SubtitleSelection:
    """Try candidates, source by source, until one passes the sync check (DESIGN.md §6.3, §6.6).
    A source is searched only when the ones before it had nothing usable, so the network is used
    only when local subtitles fail."""
    config = ctx.config
    selection = SubtitleSelection(None, [], ())
    loader = SubtitleLoader(ctx.ff, info.path, workdir, online=search.online)
    transcribe: Callable[[float, float], Sequence[Word]] | None = None
    notes: list[str] = []

    def check(cues: list[Cue], *, trusted: bool, scale: float = 1.0) -> SyncResult:
        assert transcribe is not None
        return check_sync(
            cues,
            duration=info.duration,
            transcribe=transcribe,
            config=config.analysis.sync,
            trusted=trusted,
            default_scale=scale,
        )

    for source in search.sources:
        with ctx.ui.status(f"Looking for subtitles: {source}"):
            found = search.search(source)
        notes += found.notes
        if source == "embedded":
            loader.embedded = [c for c in found.candidates if c.stream is not None]
        for candidate in found.candidates:
            entry = _candidate_report(candidate)
            selection.tried.append(entry)
            try:
                if candidate.source == "embedded" and loader.pending:
                    with ctx.ui.progress("Reading subtitles", info.duration) as update:
                        loader.on_progress = update
                        text = loader.text(candidate)
                else:
                    with ctx.ui.status(f"Loading {candidate.label}"):
                        text = loader.text(candidate)
                cues = parse_subtitles(text, fps=info.frame_rate)
            except SubtitleError as exc:
                entry["result"] = f"unusable: {exc}"
                ctx.ui.info(f"subtitles {candidate.label}: {entry['result']}")
                continue
            entry["cues"] = len(cues)
            if not cues:
                entry["result"] = "no dialogue cues"
                ctx.ui.info(f"subtitles {candidate.label}: {entry['result']}")
                continue
            scale = frame_rate_ratio(candidate, text, info.frame_rate)
            if scale != 1.0:
                entry["frame_rate_ratio"] = scale
            if transcribe is None:
                # Loading the model shows its own status line, so not inside the one below.
                transcribe = anchor_transcriber(ctx, audio)
            with ctx.ui.status(f"Checking the sync of {candidate.label}"):
                sync = check(cues, trusted=candidate.trusted, scale=scale)
            if (
                not sync.passed
                and sync.problem in ("unmatched", "error")
                and config.analysis.sync.ffsubsync == "fallback"
                and ffsubsync.ffsubsync_command() is not None
            ):
                entry["first_sync"] = _sync_report(sync) | {"reason": sync.reason}
                try:
                    with ctx.ui.status(f"Re-syncing {candidate.label} with ffsubsync"):
                        resynced = ffsubsync.resync(
                            info.path,
                            stream_index,
                            text,
                            workdir,
                            ffmpeg=ctx.ff.ffmpeg,
                            fps=info.frame_rate,
                            language=candidate.language,
                        )
                        resynced_cues = parse_subtitles(resynced)
                        # ffsubsync's output must be in sync: search only the trusted few seconds.
                        second = check(resynced_cues, trusted=True)
                    entry["ffsubsync"] = "in sync" if second.passed else f"still out of sync: {second.reason}"
                    if second.passed:
                        text, cues, sync = resynced, resynced_cues, second
                except SubtitleError as exc:
                    entry["ffsubsync"] = f"failed: {exc}"
            entry["sync"] = _sync_report(sync)
            entry["result"] = "used" if sync.passed else sync.reason
            if sync.passed:
                selection.chosen = SubtitleChoice(candidate, text, cues, sync)
                selection.notes = tuple(notes)
                return selection
            ctx.ui.info(f"subtitles {candidate.label}: not usable: {sync.reason} ({describe_sync(sync)})")
    selection.notes = tuple(notes)
    return selection


def no_subtitles_reason(selection: SubtitleSelection) -> str:
    if selection.tried:
        return "no usable subtitles: " + "; ".join(f"{e['label']}: {e['result']}" for e in selection.tried)
    notes = f" ({'; '.join(selection.notes)})" if selection.notes else ""
    return f"no subtitles found{notes}"


def _read_all(audio: AudioSource, windows: Sequence[Window]) -> list[Audio]:
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        return list(pool.map(lambda w: audio.read(w.start, w.end), windows))


def transcribe_windows(
    ctx: Context,
    transcriber: Transcriber,
    audio: AudioSource,
    windows: Sequence[Window],
    duration: float,
    label: str,
) -> tuple[list[Detection], int]:
    """Transcribe the windows (trimmed to their speech) and find listed words in them."""
    with ctx.ui.status("Reading audio"):
        samples = _read_all(audio, windows)
        trimmed = [
            trim_to_speech(Clip(w.start, a), ctx.detect_speech) for w, a in zip(windows, samples, strict=True)
        ]
    kept = [
        (w, clip, (head, tail))
        for w, (clip, head, tail) in zip(windows, trimmed, strict=True)
        if clip.duration >= MIN_CLIP_S
    ]
    if not kept:
        return [], 0
    clips = [clip for _, clip, _ in kept]
    effective = [Window(c.start, c.start + c.duration, w.reasons, w.cues) for w, c, _ in kept]
    with ctx.ui.progress(label, sum(c.duration for c in clips)) as update:
        transcripts = transcriber.transcribe(
            clips, language=ctx.config.analysis.language, prompt=ctx.prompt, vad=False, on_progress=update
        )
    return detect_in_windows(ctx.lexicon, effective, transcripts, duration, [edges for _, _, edges in kept])


def _flag_counts(flags: Sequence[FlaggedCue]) -> dict[str, int]:
    return {reason: sum(reason in f.reasons for f in flags) for reason in ("lexicon", "masked", "hint")}


def analyse(
    ctx: Context,
    strategy: str,
    info: MediaInfo,
    audio: AudioSource,
    stream_index: int,
    workdir: Path,
    *,
    explicit: Path | None = None,
    track: Audio | None = None,
    lap: Callable[[str], None] = lambda name: None,
) -> GuidedResult:
    """Run a subtitle-guided strategy. `track` is the decoded soundtrack, which `hybrid` needs."""
    config = ctx.config
    settings = config.analysis.targeted
    duration = info.duration
    search = SubtitleSearch(info, config.subtitles, explicit=explicit, online=ctx.online(info))
    selection = select_subtitles(ctx, info, audio, stream_index, workdir, search)
    lap("subtitles")
    report: dict[str, Any] = {
        "subtitle_candidates": selection.tried,
        "subtitle_search": list(selection.notes),
    }
    if search.online is not None and "opensubtitles" in search.searched:
        report["opensubtitles"] = search.online.report
    if selection.chosen is None:
        raise Fallback(no_subtitles_reason(selection), report)
    choice = selection.chosen
    sync = choice.sync.model
    report["subtitle"] = _candidate_report(choice.candidate) | {
        "cues": len(choice.cues),
        "sync": _sync_report(choice.sync),
    }
    ctx.ui.info(
        f"Subtitles: {choice.candidate.label} · {len(choice.cues)} cues · {describe_sync(choice.sync)}"
    )

    flags = flag_cues(ctx.lexicon, choice.cues)
    pad = search_padding(settings.window_padding_s, sync)
    spans = flagged_windows(flags, sync, pad)
    uncovered: list[tuple[float, float]] = []
    if strategy == "hybrid":
        assert track is not None
        with ctx.ui.progress("Finding speech", duration) as update:
            speech = ctx.detect_speech(track, update)
        uncovered = uncovered_speech(speech, choice.cues, sync)
        spans += [Window(start, end, frozenset({"uncovered"})) for start, end in uncovered]
        lap("vad")

    def plan(raw: Sequence[Window]) -> list[Window]:
        return plan_windows(
            raw,
            duration=duration,
            min_window=settings.min_window_s,
            max_window=settings.max_window_s,
            merge_gap=settings.merge_gap_s,
        )

    windows = plan(spans)
    seconds = audio_seconds(windows)
    coverage = seconds / duration if duration else 0.0
    counts = _flag_counts(flags)
    report["windows"] = {
        "flagged_cues": counts,
        "uncovered_regions": len(uncovered),
        "count": len(windows),
        "audio_seconds": round(seconds, 1),
        "coverage": round(coverage, 4),
    }
    ctx.ui.info(
        f"Plan: {len(flags)} flagged cues ({counts['lexicon']} listed, {counts['masked']} masked, "
        f"{counts['hint']} hint)"
        + (f" + {len(uncovered)} unsubtitled speech regions" if strategy == "hybrid" else "")
        + f" → {len(windows)} windows · {seconds / 60:.1f} min of audio ({coverage:.1%})"
    )
    if coverage > settings.max_coverage and config.analysis.fallback_to_full:
        raise Fallback(
            f"the windows would cover {coverage:.0%} of the runtime (more than max_coverage, "
            f"{settings.max_coverage:.0%}), so transcribing everything is cheaper",
            report,
        )

    detections: list[Detection] = []
    words = 0
    model: ModelChoice | None = None
    if windows:
        model, transcriber = ctx.transcriber(strategy)
        detections, words = transcribe_windows(ctx, transcriber, audio, windows, duration, "Transcribing")
        confirmed, unconfirmed = split_confirmed(flags, detections, sync, pad)
        if unconfirmed and settings.expand_by_s > 0:
            grow = settings.expand_by_s
            wider = plan(
                [
                    Window(w.start - grow, w.end + grow, frozenset({"expanded"}), w.cues)
                    for w in flagged_windows(unconfirmed, sync, pad)
                ]
            )
            report["windows"]["expanded"] = len(wider)
            more, heard = transcribe_windows(ctx, transcriber, audio, wider, duration, "Re-checking")
            detections, words = dedupe(detections + more), words + heard
        lap("transcribe")
    confirmed, unconfirmed = split_confirmed(flags, detections, sync, pad)
    strong = len(confirmed) + len(unconfirmed)
    report["confirmation"] = {"strong_flags": strong, "confirmed": len(confirmed)}
    if strong >= MIN_FLAGS_TO_ESCALATE and 2 * len(confirmed) < strong and config.analysis.fallback_to_full:
        raise Fallback(
            f"speech recognition confirmed only {len(confirmed)} of {strong} flagged cues, "
            "so the subtitles do not match this audio",
            report,
        )

    resolution = settings.on_unconfirmed
    for flag in unconfirmed:
        detections += resolve_unconfirmed(flag, resolution, sync, duration)
    report["unconfirmed"] = [
        {"cue": f.cue.index, "text": f.cue.text, "resolution": resolution} for f in unconfirmed
    ]
    detections = attribute(dedupe(detections), flags, sync, pad)
    return GuidedResult(detections, words, model, report)
