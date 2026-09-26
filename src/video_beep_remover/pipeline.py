"""Run the stages for one file (DESIGN.md §5.1) and choose between strategies (§7)."""

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from video_beep_remover import __version__
from video_beep_remover import guided as guided_analysis
from video_beep_remover.asr.base import Clip, Transcriber, build_prompt
from video_beep_remover.asr.cache import VERSION as TRANSCRIPT_VERSION
from video_beep_remover.asr.cache import TranscriptCache, TranscriptStore, settings_key
from video_beep_remover.asr.faster_whisper import (
    FasterWhisperTranscriber,
    ModelChoice,
    resolve_anchor_model,
    resolve_model,
)
from video_beep_remover.asr.vad import Regions, SpeechDetector, silero_speech
from video_beep_remover.config.loader import LoadedConfig, cache_root, config_hash
from video_beep_remover.detect.intervals import build_intervals
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.detect.matcher import detect_in_words
from video_beep_remover.detect.refine import refine_edges
from video_beep_remover.errors import ConfigError, SubtitleError, UsageError, VbrError
from video_beep_remover.media.audio import (
    SAMPLE_RATE,
    ArrayAudioSource,
    Audio,
    AudioSource,
    SeekingAudioSource,
    decode_track,
    read_window,
)
from video_beep_remover.media.dialogue import same_dialogue
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import MediaInfo, StreamInfo, probe, select_audio_stream
from video_beep_remover.media.render import CENSORED_TAG, RenderResult, StreamPlan, plan_streams, render
from video_beep_remover.models import CensorInterval, Detection
from video_beep_remover.report import (
    SCHEMA_VERSION,
    detection_dict,
    edl_text,
    interval_dict,
    read_report,
    report_detections,
    report_intervals,
    review_srt,
    write_json,
    write_text,
)
from video_beep_remover.subtitles.cache import SubtitleCache
from video_beep_remover.subtitles.online import OnlineSubtitles
from video_beep_remover.subtitles.opensubtitles import OpenSubtitlesClient
from video_beep_remover.subtitles.oshash import fingerprint, opensubtitles_hash
from video_beep_remover.subtitles.output import censor_streams, write_censored_copy
from video_beep_remover.ui import UI, NullUI, Progress

__all__ = ["UI", "FileResult", "Job", "NullUI", "Pipeline", "Progress", "RunOptions", "resolve_output"]

Status = Literal["cleaned", "copied", "clean", "scanned", "skipped"]
TranscriberFactory = Callable[[ModelChoice], Transcriber]


@dataclass(frozen=True)
class RunOptions:
    dry_run: bool = False
    output: Path | None = None  # -o: a file (single input) or a directory
    report: Path | None = None  # --report: a file (single input) or a directory
    edl: bool = False
    review_srt: bool = False
    overwrite: bool = False
    skip_existing: bool = False
    keep_temp: bool = False
    subtitles: Path | None = None  # --subtitles: use this file, skip the search


@dataclass
class FileResult:
    input: Path
    status: Status
    output: Path | None = None
    report: Path | None = None
    edl: Path | None = None
    review: Path | None = None  # the review subtitles
    subtitle_copy: Path | None = None  # the censored copy of the subtitle file the analysis used
    detections: int = 0
    intervals: int = 0
    strategy: str = ""  # the strategy that ran ("report" for vbr render)
    notes: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    strategy: str  # the one that ran
    fallback_reason: str | None
    detections: list[Detection]
    words: int
    model: ModelChoice | None
    report: dict[str, Any]  # strategy-specific report sections
    from_cache: Literal["all", "some", "none"] | None = None  # transcripts; None: nothing to transcribe


@dataclass
class Job:
    """One file between analysis and output: what is left to render and write. It owns its temporary
    directory, which Pipeline.finish deletes, so the render can run after the next file's analysis
    has started (batch mode)."""

    source: Path
    options: RunOptions
    many: bool
    info: MediaInfo
    stream: StreamInfo
    workdir: Path
    result: FileResult
    output: Path | None  # None for a dry run
    started: float
    clock: float
    intervals: list[CensorInterval] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    subtitle: dict[str, Any] | None = None  # the report's "subtitle": what the analysis used
    report: dict[str, Any] | None = None  # None: no report is written (vbr render)
    render: bool = False
    rendered: RenderResult | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def lap(self, name: str) -> None:
        now = time.monotonic()
        self.timings[name] = round(self.timings.get(name, 0.0) + now - self.clock, 3)
        self.clock = now


def resolve_output(source: Path, template: str, explicit: Path | None, *, many: bool) -> Path:
    """Apply output.path ({stem}, {ext}, {dir}); -o may name a file or a directory.

    An -o path without a file extension is a directory, even if it does not exist yet."""
    try:
        name = template.format(stem=source.stem, ext=source.suffix, dir=str(source.parent))
    except (KeyError, IndexError, ValueError) as exc:
        raise ConfigError(
            f"output.path {template!r}: unknown placeholder {exc} (use {{stem}}, {{ext}}, {{dir}})"
        ) from exc
    candidate = Path(name).expanduser()
    if explicit is not None:
        if many or explicit.is_dir() or not explicit.suffix or str(explicit).endswith(("/", "\\")):
            return explicit / candidate.name
        return explicit
    return candidate if candidate.is_absolute() else source.parent / candidate


def _report_path(options: RunOptions, source: Path, output: Path | None, many: bool) -> Path:
    if options.report is not None:
        if many or options.report.is_dir():
            return options.report / f"{source.stem}.vbr.json"
        return options.report
    base = output or source
    return base.with_name(f"{base.stem}.vbr.json")


class _Track:
    """The decoded soundtrack, decoded at most once per file and shared by every stage that needs it."""

    def __init__(
        self, pipeline: "Pipeline", source: Path, info: MediaInfo, stream: StreamInfo, workdir: Path
    ):
        self.pipeline = pipeline
        self.source = source
        self.info = info
        self.stream = stream
        self.path = workdir / "audio.f32"
        self.audio: Audio | None = None

    def get(self) -> Audio:
        if self.audio is None:
            with self.pipeline.ui.progress("Decoding audio", self.info.duration) as update:
                self.audio = decode_track(
                    self.pipeline.ff, self.source, self.stream.index, self.path, on_progress=update
                )
        return self.audio

    def release(self) -> None:
        self.audio = None  # closes the memory map, so the temp dir can be deleted (Windows)


@dataclass
class _CacheScope:
    """What the transcript cache needs to know about the file being analysed."""

    fingerprint: str
    stream: int
    duration: float
    speech: Regions | None  # cached speech regions, if any
    stores: dict[str, TranscriptStore] = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        ui: UI | None = None,
        ff: FFmpeg | None = None,
        categories: list[str] | None = None,
        transcriber_factory: TranscriberFactory | None = None,
        speech_detector: SpeechDetector | None = None,
        opensubtitles: OpenSubtitlesClient | None = None,
    ) -> None:
        self.config = loaded.config
        self.ui: UI = ui or NullUI()
        self.ff = ff or FFmpeg(self.config.tools.ffmpeg, self.config.tools.ffprobe)
        self.lexicon = compile_lexicon(self.config.lexicon, only=categories, base_dir=loaded.base_dir)
        for warning in self.lexicon.warnings:
            self.ui.warn(warning)
        if not self.lexicon.terms and not self.lexicon.masked_patterns:
            raise ConfigError("the word list is empty: enable a category or add terms")
        self.prompt = build_prompt(self.config.transcription.initial_prompt, self.lexicon)
        self._factory = transcriber_factory or self._load_transcriber
        self._transcribers: dict[ModelChoice, Transcriber] = {}
        self.detect_speech: SpeechDetector = speech_detector or silero_speech
        self.subtitle_cache = SubtitleCache(cache_root(self.config))
        self._opensubtitles = opensubtitles
        self.tag = f"{__version__};{config_hash(self.config)}"  # VBR_CENSORED on every output
        self.transcript_cache = TranscriptCache(cache_root(self.config))
        self._scope: _CacheScope | None = None  # the file being analysed, for the transcript cache

    def opensubtitles(self) -> OpenSubtitlesClient | None:
        """The OpenSubtitles client, when there is a key and the network may be used. One client
        serves a whole batch, so it logs in at most once."""
        settings = self.config.subtitles.opensubtitles
        if self.config.offline or not settings.enabled or not settings.api_key:
            return None
        if self._opensubtitles is None:
            self._opensubtitles = OpenSubtitlesClient(
                settings.api_key,
                user_agent=settings.user_agent,
                username=settings.username,
                password=settings.password,
            )
        return self._opensubtitles

    def online_source(self, info: MediaInfo) -> OnlineSubtitles | None:
        if "opensubtitles" not in self.config.subtitles.sources:
            return None
        return OnlineSubtitles(self.config, info, client=self.opensubtitles(), cache=self.subtitle_cache)

    def _load_transcriber(self, choice: ModelChoice) -> Transcriber:
        settings = self.config.transcription
        whisper = FasterWhisperTranscriber(
            choice,
            beam_size=settings.beam_size,
            batch_size=settings.batch_size,
            vad_filter=settings.vad_filter,
            offline=self.config.offline,
        )
        if not choice.align:
            return whisper
        from video_beep_remover.asr.whisperx import WhisperXTranscriber

        return WhisperXTranscriber(
            whisper,
            language=self.config.analysis.language,
            align_model=settings.align_model,
            offline=self.config.offline,
        )

    def model_choice(self, role: str) -> ModelChoice:
        """The model for a role: "full", "targeted" or "hybrid" (by strategy), or "anchor"."""
        settings, language = self.config.transcription, self.config.analysis.language
        if role == "anchor":
            return resolve_anchor_model(settings, language=language)
        return resolve_model(settings, strategy=role, language=language)

    def transcriber(self, role: str) -> tuple[ModelChoice, Transcriber]:
        """The model for a role, loaded on first use. Loading is announced on a line of its own rather
        than a live status, since it can happen while one is shown (e.g. during the sync check)."""
        choice = self.model_choice(role)
        if choice not in self._transcribers:
            self.ui.info(f"Loading Whisper model {choice.describe()}")
            self._transcribers[choice] = self._factory(choice)
        return choice, self._transcribers[choice]

    def transcripts(self, role: str) -> TranscriptStore | None:
        """The cached transcripts of the file being analysed, for the model and settings of `role`."""
        scope = self._scope
        if scope is None:
            return None
        choice = self.model_choice(role)
        settings = self.config.transcription
        key = settings_key(
            version=TRANSCRIPT_VERSION,
            model=choice.name,
            compute_type=choice.compute_type,
            batched=choice.batched,
            language=self.config.analysis.language,
            beam_size=settings.beam_size,
            # The prompt setting rather than the prompt itself: "auto" words it from the word list,
            # and editing the list should not make everything be transcribed again.
            prompt=None if role == "anchor" else settings.initial_prompt,
            vad_filter=settings.vad_filter,
            # Only when aligning, so that turning alignment on left the other keys as they were.
            **({"align_model": settings.align_model} if choice.align else {}),
        )
        if key not in scope.stores:
            scope.stores[key] = self.transcript_cache.store(
                scope.fingerprint, scope.stream, choice.name, key, scope.duration
            )
        return scope.stores[key]

    def guided_context(self) -> guided_analysis.Context:
        return guided_analysis.Context(
            config=self.config,
            lexicon=self.lexicon,
            prompt=self.prompt,
            ui=self.ui,
            ff=self.ff,
            transcriber=self.transcriber,
            detect_speech=self.detect_speech,
            online=self.online_source,
            transcripts=self.transcripts,
            model_choice=self.model_choice,
        )

    def _full(
        self, track: _Track, lap: Callable[[str], None], fallback_reason: str | None, report: dict[str, Any]
    ) -> Analysis:
        store = self.transcripts("full")
        words = store.track() if store else None
        cached = words is not None
        if words is None:
            audio = track.get()
            lap("decode")
            _, transcriber = self.transcriber("full")
            with self.ui.progress("Transcribing", len(audio) / SAMPLE_RATE) as update:
                [words] = transcriber.transcribe(
                    [Clip(0.0, audio)],
                    language=self.config.analysis.language,
                    prompt=self.prompt,
                    vad=self.config.transcription.vad_filter,
                    on_progress=update,
                )
            if store:
                store.add_track(words, len(audio) / SAMPLE_RATE)
        lap("transcribe")
        return Analysis(
            "full",
            fallback_reason,
            detect_in_words(self.lexicon, words),
            len(words),
            self.model_choice("full"),
            report,
            "all" if cached else "none",
        )

    def _track_speech(self, track: _Track, audio: Audio | None) -> Callable[[Progress], Regions]:
        """Speech in the whole track, for hybrid: from the cache, or found with VAD and then cached."""
        scope = self._scope

        def find(update: Progress) -> Regions:
            if scope is not None and scope.speech is not None:
                return scope.speech
            regions = self.detect_speech(audio if audio is not None else track.get(), update)
            if scope is not None:
                self.transcript_cache.remember_speech(scope.fingerprint, scope.stream, regions)
            return regions

        return find

    def _analyse(
        self,
        source: Path,
        info: MediaInfo,
        stream: StreamInfo,
        workdir: Path,
        track: _Track,
        lap: Callable[[str], None],
        subtitles: Path | None = None,
    ) -> Analysis:
        """Run the configured strategy, falling back to `full` where §7 says so."""
        analysis = self.config.analysis
        requested = analysis.strategy
        if requested == "full":
            if subtitles is not None:
                self.ui.warn("--subtitles is not used with the full strategy")
            return self._full(track, lap, None, {})
        audio: AudioSource = SeekingAudioSource(self.ff, source, stream.index)
        speech = None
        if requested == "hybrid":
            decoded = None
            if self._scope is None or self._scope.speech is None:
                decoded = track.get()  # VAD runs over the whole track, so windows slice it too
                audio = ArrayAudioSource(decoded)
                lap("decode")
            speech = self._track_speech(track, decoded)
        try:
            found = guided_analysis.analyse(
                self.guided_context(),
                requested,
                info,
                audio,
                stream.index,
                workdir,
                explicit=subtitles,
                speech=speech,
                lap=lap,
            )
        except guided_analysis.Fallback as fallback:
            if not analysis.fallback_to_full:
                raise VbrError(
                    f"strategy {requested!r} could not run: {fallback.reason}; fallback_to_full is false"
                ) from fallback
            self.ui.info(f"using the full strategy instead of {requested!r}: {fallback.reason}")
            return self._full(track, lap, fallback.reason, fallback.report)
        windows = found.report.get("windows", {})
        total = windows.get("count", 0) + windows.get("expanded", 0)
        cached, partly = windows.get("cached", 0), windows.get("partly_cached", 0)
        from_cache: Literal["all", "some", "none"] | None = None
        if total:
            from_cache = "all" if cached == total else "some" if cached or partly else "none"
        return Analysis(requested, None, found.detections, found.words, found.model, found.report, from_cache)

    def process(
        self, source: Path, options: RunOptions, *, many: bool = False, from_folder: bool = False
    ) -> FileResult:
        """Analyse one file, then render it and write the report and other outputs."""
        prepared = self.prepare(source, options, many=many, from_folder=from_folder)
        return prepared if isinstance(prepared, FileResult) else self.finish(prepared)

    def _check_output(self, source: Path, options: RunOptions, many: bool) -> Path | FileResult:
        """The output path, or a skipped result for --skip-existing."""
        output = resolve_output(source, self.config.output.path, options.output, many=many)
        if output.resolve() == source.resolve():
            raise UsageError(f"the output would overwrite the input: {output}")
        if output.exists() and not (options.overwrite or self.config.output.overwrite):
            if options.skip_existing:
                return FileResult(source, "skipped", output=output, notes=["output already exists"])
            raise UsageError(f"{output} already exists (use --overwrite or --skip-existing)")
        return output

    def prepare(
        self, source: Path, options: RunOptions, *, many: bool = False, from_folder: bool = False
    ) -> "Job | FileResult":
        """Probe and analyse one file. Returns the job that renders it and writes its outputs, or the
        result straight away for a skipped file. `from_folder`: the file was found by searching a
        folder, so a file vbr already censored is skipped rather than censored again."""
        cfg = self.config
        started = time.monotonic()
        if options.subtitles is not None and not options.subtitles.is_file():
            raise UsageError(f"subtitle file not found: {options.subtitles}")
        output = None
        if not options.dry_run:
            checked = self._check_output(source, options, many)
            if isinstance(checked, FileResult):
                return checked
            output = checked

        info = probe(self.ff, source)
        tag = info.tags.get(CENSORED_TAG.lower())
        if tag is not None:
            note = f"already censored by vbr ({CENSORED_TAG}={tag})"
            if from_folder:
                return FileResult(source, "skipped", notes=[note])
            self.ui.warn(f"{source.name} is {note}")
        stream = select_audio_stream(info, cfg.analysis.language, cfg.analysis.audio_stream)
        self.ui.info(f"{source.name}: {info.duration / 60:.1f} min, analysing audio {stream.describe()}")

        job = Job(
            source=source,
            options=options,
            many=many,
            info=info,
            stream=stream,
            workdir=Path(tempfile.mkdtemp(prefix="vbr-")),
            result=FileResult(source, "scanned"),
            output=output,
            started=started,
            clock=started,
        )
        job.lap("probe")
        try:
            self._analyse_job(job)
        except BaseException:
            self._cleanup(job, self.ui)
            raise
        return job

    def _analyse_job(self, job: Job) -> None:
        cfg = self.config
        source, info, stream, result = job.source, job.info, job.stream, job.result
        track = _Track(self, source, info, stream, job.workdir)
        found = fingerprint(source) if cfg.cache.transcripts else None
        if found is not None:
            speech = self.transcript_cache.speech(found, stream.index)
            self._scope = _CacheScope(found, stream.index, info.duration, speech)
        try:
            analysis = self._analyse(source, info, stream, job.workdir, track, job.lap, job.options.subtitles)
            detections = analysis.detections
            intervals = build_intervals(
                detections,
                duration=info.duration,
                pad_before=cfg.censor.pad_before_ms / 1000,
                pad_after=cfg.censor.pad_after_ms / 1000,
                min_duration=cfg.censor.min_duration_ms / 1000,
                merge_gap=cfg.censor.merge_gap_ms / 1000,
            )
            if cfg.censor.refine_edges and intervals:
                audio: AudioSource = (
                    ArrayAudioSource(track.audio)
                    if track.audio is not None
                    else SeekingAudioSource(self.ff, source, stream.index)
                )
                intervals = refine_edges(
                    intervals, audio, duration=info.duration, merge_gap=cfg.censor.merge_gap_ms / 1000
                )
                job.lap("refine")
        finally:
            self._scope = None
            track.release()
            if not job.options.keep_temp:
                track.path.unlink(missing_ok=True)  # the decoded track is not needed for rendering
            if found is not None:
                self.transcript_cache.evict(int(cfg.cache.max_size_gb * 1024**3))
        result.detections, result.intervals = len(detections), len(intervals)
        result.strategy = analysis.strategy
        heard = sum(d.source == "asr" for d in detections)
        estimated = len(detections) - heard
        self.ui.info(
            f"{len(detections)} listed words found"
            + (f" ({heard} heard, {estimated} from subtitles only)" if estimated else "")
            + f" → {len(intervals)} spans to mute"
        )

        model = analysis.model
        job.report = {
            "schema_version": SCHEMA_VERSION,
            "tool_version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "input": {
                "path": str(source),
                "size": info.size,
                "oshash": opensubtitles_hash(source),
                "duration": round(info.duration, 3),
            },
            "audio_stream": {
                "index": stream.index,
                "codec": stream.codec,
                "channels": stream.channels,
                "language": stream.language,
            },
            "strategy": {
                "requested": cfg.analysis.strategy,
                "used": analysis.strategy,
                "fallback_reason": analysis.fallback_reason,
            },
            **analysis.report,
            "transcription": {
                "backend": ("whisperx" if model.align else "faster-whisper") if model else None,
                "model": model.name if model else None,
                "device": model.device if model else None,
                "compute_type": model.compute_type if model else None,
                "prompt": self.prompt,
                "words": analysis.words,
                "from_cache": analysis.from_cache,
            },
            "categories": list(self.lexicon.categories),
            "detections": [detection_dict(d) for d in detections],
            "intervals": [interval_dict(i) for i in intervals],
            "output": None,
            "timings": job.timings,
        }
        job.intervals, job.detections = intervals, detections
        subtitle = analysis.report.get("subtitle")
        job.subtitle = subtitle if isinstance(subtitle, dict) else None
        if not job.options.dry_run:
            if intervals or cfg.output.when_clean == "copy":
                job.render = True
            else:
                result.status = "clean"

    def finish(self, job: Job, ui: UI | None = None) -> FileResult:
        """Render (unless it is a dry run), write the report, EDL and review subtitles, then delete the
        job's temporary files. `ui` replaces the pipeline's own, e.g. for a render in the background."""
        ui = ui or self.ui
        try:
            if job.render:
                self._render(job, ui)
            self._write_outputs(job, ui)
        finally:
            self._cleanup(job, ui)
        return job.result

    def _render(self, job: Job, ui: UI) -> None:
        cfg = self.config
        assert job.output is not None
        plan = plan_streams(job.info, job.stream, cfg.output, cfg.analysis.language, cfg.lexicon.language)
        audio_checks: list[dict[str, Any]] = []
        if cfg.output.other_audio_streams == "auto" and job.intervals:
            with ui.status("Comparing the other audio streams"):
                plan, audio_checks = self._check_other_audio(job, plan)
        subtitles = censor_streams(
            self.ff, job.info, plan, job.workdir, self.lexicon, cfg.output.subtitle_mask, ui
        )
        plan = subtitles.plan
        for note in plan.notes:
            ui.warn(note)
        job.output.parent.mkdir(parents=True, exist_ok=True)
        with ui.progress("Rendering", job.info.duration) as update:
            rendered = render(
                self.ff,
                job.info,
                plan,
                job.intervals,
                output=job.output,
                fade=cfg.censor.fade_ms / 1000,
                output_config=cfg.output,
                workdir=job.workdir,
                subtitle_files=subtitles.files,
                tag=self.tag,
                on_progress=update,
            )
        job.lap("render")
        job.rendered = rendered
        result = job.result
        result.status = "cleaned" if job.intervals else "copied"
        result.output = job.output
        result.notes = list(plan.notes)
        output: dict[str, Any] = {
            "path": str(job.output),
            "encoders": {str(index): encoder for index, encoder in rendered.encoders.items()},
            "muted_spans": [interval_dict(i) for i in rendered.intervals],
            "verified_spans": rendered.verified_spans,
            "timeline_shift": round(rendered.timeline_shift, 3),
            "audio_checks": audio_checks,
            "subtitles": subtitles.report,
            "subtitle_copy": None,
            "notes": list(plan.notes),
        }
        used = job.subtitle or {}
        if cfg.output.subtitle_streams == "censor" and used.get("source") in ("sidecar", "explicit"):
            path = Path(str(used.get("path")))
            try:
                copy = write_censored_copy(
                    path,
                    video=job.source,
                    output=job.output,
                    language=used.get("language"),
                    lexicon=self.lexicon,
                    mask=cfg.output.subtitle_mask,
                    fps=job.info.frame_rate,
                    overwrite=job.options.overwrite or cfg.output.overwrite,
                )
                result.subtitle_copy = copy.path
                output["subtitle_copy"] = {"source": str(path), "path": str(copy.path), "masked": copy.masked}
            except SubtitleError as exc:
                ui.warn(f"no censored copy of the subtitles {path.name}: {exc}")
        if job.report is not None:
            job.report["output"] = output

    def _check_other_audio(self, job: Job, plan: StreamPlan) -> tuple[StreamPlan, list[dict[str, Any]]]:
        """other_audio_streams = "auto" gives a stream in the analysed language the same mutes; drop it
        instead if it does not carry the same dialogue (DESIGN.md §6.11)."""

        def read(index: int, start: float, end: float) -> Audio:
            return read_window(self.ff, job.source, index, start, end - start)

        checks: list[dict[str, Any]] = []
        for action in plan.actions:
            stream = action.stream
            if stream.kind != "audio" or stream.index == job.stream.index or action.action != "censor":
                continue
            check = same_dialogue(read, job.stream.index, stream.index, job.intervals, job.info.duration)
            checks.append(
                {
                    "stream": stream.index,
                    "same_dialogue": check.same,
                    "correlation": None if check.correlation is None else round(check.correlation, 3),
                    "lag": None if check.lag is None else round(check.lag, 3),
                    "compared_spans": check.compared,
                }
            )
            if not check.same:
                plan = plan.drop(
                    stream.index,
                    f"dropped audio stream {stream.describe()}: it does not carry the dialogue of the "
                    f"analysed stream ({check.describe()}), so the mutes would miss its words",
                )
        return plan, checks

    def _write_outputs(self, job: Job, ui: UI) -> None:
        cfg, options, result = self.config, job.options, job.result
        overwrite = options.overwrite or cfg.output.overwrite
        job.timings["total"] = round(time.monotonic() - job.started, 3)
        if job.report is not None and (options.report is not None or cfg.output.report):
            result.report = _report_path(options, job.source, job.output, job.many)
            write_json(result.report, job.report)
        if options.edl or cfg.output.edl:
            edl = job.source.with_suffix(".edl")
            if edl.exists() and not overwrite:
                ui.warn(f"not overwriting existing {edl.name} (use --overwrite)")
            else:
                write_text(edl, edl_text(job.intervals))
                result.edl = edl
        if options.review_srt or cfg.output.review_srt:
            rendered = job.rendered
            base = rendered.output if rendered else job.source
            review = base.with_name(f"{base.stem}.review.srt")
            if review.exists() and not overwrite and rendered is None:
                ui.warn(f"not overwriting existing {review.name} (use --overwrite)")
            else:
                spans = rendered.intervals if rendered else job.intervals
                shift = rendered.timeline_shift if rendered else 0.0
                write_text(review, review_srt(spans, job.detections, shift=shift))
                result.review = review

    def discard(self, job: Job) -> None:
        """Delete a prepared job's temporary files without rendering it."""
        self._cleanup(job, self.ui)

    def _cleanup(self, job: Job, ui: UI) -> None:
        if job.options.keep_temp:
            ui.info(f"kept temporary files in {job.workdir}")
        else:
            shutil.rmtree(job.workdir, ignore_errors=True)

    def render_report(
        self, source: Path, report: Path, options: RunOptions, *, force: bool = False
    ) -> FileResult:
        """`vbr render`: mute the report's `intervals`, which may have been edited by hand, without
        detecting anything. The report is only read, never rewritten."""
        cfg = self.config
        started = time.monotonic()
        data = read_report(report)
        intervals = report_intervals(data)
        checked = self._check_output(source, options, many=False)
        if isinstance(checked, FileResult):
            return checked
        info = probe(self.ff, source)
        mismatch = report_mismatch(data, info, source)
        if mismatch and not force:
            raise UsageError(
                f"{report.name} was made for a different file ({mismatch}); use --force to render anyway"
            )
        recorded = (data.get("audio_stream") or {}).get("index")
        requested = cfg.analysis.audio_stream
        if (
            requested == "auto"
            and isinstance(recorded, int)
            and recorded in {s.index for s in info.audio_streams}
        ):
            requested = recorded
        stream = select_audio_stream(info, cfg.analysis.language, requested)
        detections = report_detections(data)
        subtitle = data.get("subtitle")
        job = Job(
            source=source,
            options=options,
            many=False,
            info=info,
            stream=stream,
            workdir=Path(tempfile.mkdtemp(prefix="vbr-")),
            result=FileResult(
                source, "scanned", detections=len(detections), intervals=len(intervals), strategy="report"
            ),
            output=checked,
            started=started,
            clock=started,
            intervals=intervals,
            detections=detections,
            subtitle=subtitle if isinstance(subtitle, dict) else None,
            render=True,
        )
        self.ui.info(
            f"{source.name}: muting the {len(intervals)} spans in {report.name}, audio {stream.describe()}"
        )
        return self.finish(job)


def report_mismatch(data: dict[str, Any], info: MediaInfo, source: Path) -> str | None:
    """Why the report looks like it was made for another file, if it does."""
    found = data.get("input")
    recorded: dict[str, Any] = found if isinstance(found, dict) else {}
    size = recorded.get("size")
    if isinstance(size, int) and size and info.size and size != info.size:
        return f"{size} bytes, this file has {info.size}"
    oshash = recorded.get("oshash")
    if isinstance(oshash, str) and oshash and opensubtitles_hash(source) not in (None, oshash):
        return "its OpenSubtitles hash differs"
    duration = recorded.get("duration")
    if isinstance(duration, int | float) and abs(duration - info.duration) > 1.0:
        return f"{duration:.1f} s long, this file is {info.duration:.1f} s"
    return None
