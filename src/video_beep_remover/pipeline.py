"""Run the stages for one file (DESIGN.md §5.1) and choose between strategies (§7).

One file goes through Pipeline in this order:

1. prepare: where the output goes (outputs.place), the probe, the audio stream;
2. _analyse_job: _analyse runs the strategy (full here, targeted and hybrid in guided.py) and falls
   back to full; then context analysis (_run_context, context/), the muted spans (detect/intervals.py,
   detect/refine.py) and voice replacement (_replace_words, voice/);
3. finish: _render (media/render.py, subtitles/output.py), then _write_outputs (report, EDL, review
   subtitles).

batch.py calls prepare and finish separately, so that one file renders while the next is analysed.
render_report (`vbr render`) takes a report's intervals straight to finish."""

import contextlib
import dataclasses
import gc
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from video_beep_remover import __version__
from video_beep_remover import guided as guided_analysis
from video_beep_remover.asr.base import Clip, Transcriber, build_prompt
from video_beep_remover.asr.cache import VERSION as TRANSCRIPT_VERSION
from video_beep_remover.asr.cache import TranscriptCache, TranscriptStore, settings_key
from video_beep_remover.asr.faster_whisper import (
    FasterWhisperTranscriber,
    ModelChoice,
    cuda_available,
    resolve_anchor_model,
    resolve_model,
)
from video_beep_remover.asr.vad import Regions, SpeechDetector, silero_speech
from video_beep_remover.config.loader import LoadedConfig, cache_root, config_hash
from video_beep_remover.context import (
    ContextLayer,
    Line,
    ModelFactory,
    build_lines,
    check_installed,
    kept_cues,
    review_cues,
    review_labels,
    review_notes,
    verdict_dict,
    word_lines,
)
from video_beep_remover.context.analyse import ContextResult
from video_beep_remover.context.lines import COVER_MARGIN_S
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
    read_pcm_windows,
    read_window,
)
from video_beep_remover.media.dialogue import same_dialogue
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import MediaInfo, StreamInfo, probe, select_audio_stream
from video_beep_remover.media.render import (
    CENSORED_TAG,
    RenderResult,
    Splice,
    StreamPlan,
    plan_streams,
    render,
)
from video_beep_remover.models import CensorInterval, Detection, Word
from video_beep_remover.outputs import Placement, edl_path, place, report_path, review_path
from video_beep_remover.report import (
    SCHEMA_VERSION,
    detection_dict,
    edl_text,
    interval_dict,
    read_report,
    report_detection_items,
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
from video_beep_remover.subtitles.parse import parse_sounds
from video_beep_remover.ui import UI, NullUI, Progress
from video_beep_remover.voice import Candidate, Replacement, Replacer, VoiceModels
from video_beep_remover.voice import check_installed as check_voice_installed

__all__ = ["UI", "FileResult", "Job", "NullUI", "Pipeline", "Progress", "RunOptions"]

log = logging.getLogger(__name__)

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
    backup: Path | None = None  # --backup: where the unmodified original is kept
    report: Path | None = None
    edl: Path | None = None
    review: Path | None = None  # the review subtitles
    subtitle_copy: Path | None = None  # the censored copy of the subtitle file the analysis used
    detections: int = 0
    intervals: int = 0
    replaced: int = 0  # of the detections, said again as a milder word (DESIGN.md §16)
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
    subtitles: guided_analysis.SubtitleChoice | None = None  # the subtitles that guided the search
    heard: list[Word] = field(default_factory=list)  # the words heard, in time order


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
    context: dict[str, Any] | None = None  # the report's "context" section (DESIGN.md §17)
    verdicts: list[dict[str, Any] | None] = field(default_factory=list)  # one per detection
    replacements: list[Replacement] = field(default_factory=list)  # voice replacement (DESIGN.md §16)
    render: bool = False
    backup: Path | None = None  # --backup: where the original goes when the output takes its place
    rendered: RenderResult | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def spliced(self) -> bool:
        """Whether rendering adds replaced words, which it then transcribes again: the render uses the
        speech recognition model, so it must not overlap the next file's analysis (batch.py)."""
        return self.render and any(r.replaced and r.delta is not None for r in self.replacements)

    def lap(self, name: str) -> None:
        now = time.monotonic()
        self.timings[name] = round(self.timings.get(name, 0.0) + now - self.clock, 3)
        self.clock = now


def _free_memory() -> None:
    """Return the memory of dropped models: Python's, and the GPU memory PyTorch keeps cached."""
    gc.collect()
    torch = sys.modules.get("torch")  # only if a model already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _context_device(setting: str) -> str:
    """The device the context models run on: transcription.device's."""
    return ("cuda" if cuda_available() else "cpu") if setting == "auto" else setting


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
        context_models: tuple[ModelFactory, ModelFactory] | None = None,
        voice_models: tuple[Callable[[], Any], Callable[[], Any], Callable[[], Any]] | None = None,
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
        self._context_models = context_models  # (classifier, judge) factories, for tests
        self._context: ContextLayer | None = None
        self._voice_models = voice_models  # (separator, editor, speaker encoder) factories, for tests
        self._replacer: Replacer | None = None
        self._check_role = "full"  # the transcriber that checks replaced words: the analysis's

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

    def make_room(self, model: str) -> None:
        """Before `model` runs ("whisper", "context", "separator" or "editor"), drop the other large
        models, unless models.keep_loaded. Each of them can take one to several GB of GPU memory, and on
        Windows what does not fit on the GPU spills into system memory; one at a time, the peak is the
        largest of them rather than their sum. A dropped model is loaded again on its next use."""
        if self.config.models.keep_loaded:
            return
        dropped = []
        if model != "whisper" and self._transcribers:
            self._transcribers.clear()
            dropped.append("whisper")
        if model != "context" and self._context is not None and self._context.release():
            dropped.append("context")
        for name in ("separator", "editor"):
            if model != name and self._replacer is not None and self._replacer.models.release(name):
                dropped.append(name)
        if dropped:
            log.debug("freed %s before %s runs", ", ".join(dropped), model)
            _free_memory()

    def transcriber(self, role: str) -> tuple[ModelChoice, Transcriber]:
        """The model for a role, loaded on first use. Loading is announced on a line of its own rather
        than a live status, since it can happen while one is shown (e.g. during the sync check)."""
        self.make_room("whisper")
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
            heard=list(words),
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
        return Analysis(
            requested,
            None,
            found.detections,
            found.words,
            found.model,
            found.report,
            from_cache,
            found.subtitles,
            found.heard,
        )

    def process(
        self, source: Path, options: RunOptions, *, many: bool = False, from_folder: bool = False
    ) -> FileResult:
        """Analyse one file, then render it and write the report and other outputs."""
        prepared = self.prepare(source, options, many=many, from_folder=from_folder)
        return prepared if isinstance(prepared, FileResult) else self.finish(prepared)

    def _check_output(self, source: Path, options: RunOptions, many: bool) -> Placement | FileResult:
        """Where the cleaned file goes (outputs.place), or a skipped result: for --skip-existing, or for
        a file whose backup already exists, since a backup is never overwritten."""
        placement = place(source, self.config.output, options.output, many=many)
        output, backup = placement.output, placement.backup
        if backup is not None and backup.exists():
            note = f"cleaned before: its original is kept as {backup.name}"
            return FileResult(source, "skipped", output=output, backup=backup, notes=[note])
        exists = not placement.replaces_source and output.exists()
        if exists and not (options.overwrite or self.config.output.overwrite):
            if options.skip_existing:
                return FileResult(source, "skipped", output=output, notes=["output already exists"])
            raise UsageError(f"{output} already exists (use --overwrite or --skip-existing)")
        return placement

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
        if (cfg.context.enabled or cfg.replace.enabled) and self._context_models is None:
            check_installed()  # the layer runs last: fail before the transcription, not after it
        if cfg.replace.enabled and self._voice_models is None:
            check_voice_installed()
        placement = None
        if not options.dry_run:
            checked = self._check_output(source, options, many)
            if isinstance(checked, FileResult):
                return checked
            placement = checked

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
            output=placement.output if placement else None,
            started=started,
            clock=started,
            backup=placement.backup if placement else None,
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
            to_mute = detections
            context: ContextResult | None = None
            if cfg.context.enabled or cfg.replace.enabled:

                def audio_source() -> AudioSource:
                    if track.audio is not None:
                        return ArrayAudioSource(track.audio)
                    return SeekingAudioSource(self.ff, source, stream.index)

                to_mute, context = self._run_context(job, analysis, detections, audio_source)
            intervals = build_intervals(
                to_mute,
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
            if cfg.replace.enabled and context is not None:
                self._replace_words(job, analysis, detections, to_mute, intervals, context)
        finally:
            self._scope = None
            track.release()
            # Neither may hide an error from the analysis. On Windows the track cannot be deleted
            # while a view of it is still mapped; _cleanup deletes it with the rest of the job.
            with contextlib.suppress(OSError):
                if not job.options.keep_temp:
                    track.path.unlink(missing_ok=True)  # the decoded track is not needed for rendering
            with contextlib.suppress(OSError):  # e.g. another run deleting the same files
                if found is not None:
                    self.transcript_cache.evict(int(cfg.cache.max_size_gb * 1024**3))
        result.detections, result.intervals = len(detections), len(intervals)
        result.replaced = sum(r.replaced for r in job.replacements)
        result.strategy = analysis.strategy
        heard = sum(d.source == "asr" for d in detections)
        estimated = len(detections) - heard
        self.ui.info(
            f"{len(detections)} listed words found"
            + (f" ({heard} heard, {estimated} from subtitles only)" if estimated else "")
            + f" → {len(intervals)} spans to mute"
            + (f", {result.replaced} of them replaced" if result.replaced else "")
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
            "detections": [
                detection_dict(d) | ({"context": v} if v is not None else {})
                for d, v in zip(detections, job.verdicts or [None] * len(detections), strict=True)
            ],
            "intervals": [interval_dict(i) for i in intervals],
            **({"context": job.context} if job.context is not None else {}),
            **({"replacements": [r.as_dict() for r in job.replacements]} if cfg.replace.enabled else {}),
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
        except OSError as exc:  # a full disk, or a file another program holds open
            raise VbrError(f"could not write {exc.filename or 'a file'}: {exc.strerror or exc}") from exc
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
                backup=job.backup,
                fade=cfg.censor.fade_ms / 1000,
                output_config=cfg.output,
                workdir=job.workdir,
                subtitle_files=subtitles.files,
                tag=self.tag,
                on_progress=update,
                splices={
                    job.stream.index: [
                        Splice(CensorInterval(r.start, r.end), r.delta, r.delta_start)
                        for r in job.replacements
                        if r.replaced and r.delta is not None
                    ]
                },
                check_splices=self._check_splice if job.spliced else None,
            )
        job.lap("render")
        job.rendered = rendered
        result = job.result
        if rendered.unspliced:
            self._withdraw_replacements(job, dict(rendered.unspliced), ui)
        result.status = "cleaned" if job.intervals else "copied"
        result.output = job.output
        result.backup = job.backup
        result.notes = list(plan.notes)
        output: dict[str, Any] = {
            "path": str(job.output),
            "backup": str(job.backup) if job.backup else None,
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
                    backup=job.backup,
                )
                result.subtitle_copy = copy.path
                output["subtitle_copy"] = {"source": str(path), "path": str(copy.path), "masked": copy.masked}
            except SubtitleError as exc:
                ui.warn(f"no censored copy of the subtitles {path.name}: {exc}")
        if job.report is not None:
            job.report["output"] = output

    def _check_splice(self, audio: Audio, start: float, span: CensorInterval) -> str | None:
        """Why a replaced word fails in the rendered file: a listed word is heard in its span."""
        words = self._check_transcriber(audio, start)
        found = detect_in_words(self.lexicon, words)
        leaked = [d for d in found if d.start < span.end and span.start < d.end]
        return f"a listed word is heard in the output: {leaked[0].heard.strip()!r}" if leaked else None

    def _withdraw_replacements(self, job: Job, failed: dict[CensorInterval, str], ui: UI) -> None:
        """Record that the renderer muted replaced words that failed in the output (verify_splices)."""
        withdrawn = []
        for replacement in job.replacements:
            reason = failed.get(CensorInterval(replacement.start, replacement.end))
            if replacement.replaced and reason is not None:
                replacement = dataclasses.replace(replacement, replaced=False, reason=reason, delta=None)
            withdrawn.append(replacement)
        job.replacements = withdrawn
        job.result.replaced = sum(r.replaced for r in withdrawn)
        if job.report is not None and "replacements" in job.report:
            job.report["replacements"] = [r.as_dict() for r in withdrawn]
        ui.warn(f"{len(failed)} replaced words failed the check of the output and are muted instead")

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
            result.report = report_path(options.report, job.source, job.output, many=job.many)
            write_json(result.report, job.report)
        if options.edl or cfg.output.edl:
            edl = edl_path(job.source, job.backup)
            if edl.exists() and not overwrite:
                ui.warn(f"not overwriting existing {edl.name} (use --overwrite)")
            else:
                write_text(edl, edl_text(job.intervals))
                result.edl = edl
        if options.review_srt or cfg.output.review_srt:
            rendered = job.rendered
            base = rendered.output if rendered else job.source
            review = review_path(base)
            if review.exists() and not overwrite and rendered is None:
                ui.warn(f"not overwriting existing {review.name} (use --overwrite)")
            else:
                spans = rendered.intervals if rendered else job.intervals
                shift = rendered.timeline_shift if rendered else 0.0
                notes = review_notes(job.verdicts)
                extra = review_cues(job.context, spans) + kept_cues(job.detections, job.verdicts, spans)
                labels = review_labels(job.context)
                replaced = [
                    (r.start, r.end, f"{r.word} → {r.substitute}") for r in job.replacements if r.replaced
                ]
                write_text(
                    review,
                    review_srt(
                        spans,
                        job.detections,
                        shift=shift,
                        notes=notes,
                        extra=extra,
                        labels=labels,
                        replaced=replaced,
                    ),
                )
                result.review = review

    def discard(self, job: Job) -> None:
        """Delete a prepared job's temporary files without rendering it."""
        self._cleanup(job, self.ui)

    def _cleanup(self, job: Job, ui: UI) -> None:
        if job.options.keep_temp:
            ui.info(f"kept temporary files in {job.workdir}")
        else:
            shutil.rmtree(job.workdir, ignore_errors=True)

    def context_layer(self) -> ContextLayer:
        """The context layer (DESIGN.md §17), created once: its models serve every file of a batch."""
        if self._context is None:
            device = _context_device(self.config.transcription.device)
            classifier, judge = self._context_models or (None, None)
            self._context = ContextLayer(
                self.config,
                device=device,
                cache_dir=cache_root(self.config) / "context",
                classifier_factory=classifier,
                judge_factory=judge,
            )
            settings = self.config.context
            if settings.harmless == "keep" or settings.sexual == "mute":
                self.ui.warn(
                    "acting on context verdicts is experimental: it is measured on a small labelled set "
                    "only (DESIGN.md §17.7); check the review subtitles (--review-srt)"
                )
            if settings.judge == "api" and self._context.judge_name is not None:
                from video_beep_remover.context.api import endpoint

                host = httpx.URL(endpoint(settings.api)[0]).host
                self.ui.warn(
                    f"the context judge is {self._context.judge_name}: each line it is asked about is "
                    f"sent to {host}, with its neighbours"
                )
            if settings.harmless == "keep" and self._context.judge_name is None:
                self.ui.warn('context.harmless = "keep" keeps nothing without a judge (context.judge)')
        return self._context

    def _run_context(
        self,
        job: Job,
        analysis: Analysis,
        detections: list[Detection],
        audio: Callable[[], AudioSource],
    ) -> tuple[list[Detection], ContextResult]:
        """Context verdicts for the report and the review subtitles (DESIGN.md §17). Returns what to mute
        (the detections, less the uses kept as harmless with context.harmless = "keep", plus a span for
        each line flagged as sexual with context.sexual = "mute"), and the verdicts."""
        chosen = analysis.subtitles
        if chosen is not None:
            try:
                sounds = parse_sounds(chosen.text, fps=job.info.frame_rate)
            except SubtitleError:
                sounds = []
            lines = build_lines(chosen.cues, sounds, chosen.sync.model, analysis.heard)
        else:
            lines = word_lines(analysis.heard)
        layer = self.context_layer()
        self.make_room("context")
        with self.ui.status("Reading the dialogue in context"):
            result, section = layer.run(detections, lines, analysis.heard)
        job.context = section
        job.verdicts = [verdict_dict(v, result.lines) for v in result.verdicts]
        settings = self.config.context
        keep = settings.harmless == "keep"
        to_mute = [
            d for d, v in zip(detections, result.verdicts, strict=True) if not (keep and v.use == "harmless")
        ]
        actions = []
        if keep:
            section["kept"] = len(detections) - len(to_mute)
            actions.append(f"{section['kept']} kept as harmless")
        if settings.sexual == "mute":
            flagged = [(i, s) for i, s in enumerate(result.sexual) if s.certain and s.line.text]
            spans = self._line_spans([found.line for _, found in flagged], analysis, audio, job.info.duration)
            for (i, found), (start, end, heard) in zip(flagged, spans, strict=True):
                section["sexual_lines"][i]["muted"] = {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "from": "heard" if heard else "cue",
                }
                to_mute.append(
                    Detection(
                        start,
                        end,
                        found.line.text,
                        "sexual line",
                        "context",
                        1.0,
                        "asr" if heard else "cue",
                        found.line.cue,
                    )
                )
            actions.append(f"{len(flagged)} sexual lines muted")
        counts = section["verdicts"]
        certain = sum(item["certain"] for item in section["sexual_lines"])
        self.ui.info(
            f"Context{'' if actions else ' (report only)'}: {counts['profane']} profane, "
            f"{counts['harmless']} probably harmless, {counts['unsure']} unsure · {certain} sexual lines"
            + (
                f" (+{len(section['sexual_lines']) - certain} possible)"
                if len(section["sexual_lines"]) > certain
                else ""
            )
            + "".join(f" · {action}" for action in actions)
            + ("" if layer.judge_name else " · no judge: " + section.get("judge_off", "off"))
        )
        job.lap("context")
        return to_mute, result

    def _line_spans(
        self,
        lines: Sequence[Line],
        analysis: Analysis,
        audio: Callable[[], AudioSource],
        duration: float,
    ) -> list[tuple[float, float, bool]]:
        """Where each line is spoken, from its first heard word to its last, and whether words were heard
        there at all; a line nothing was heard in keeps its own span. A subtitle line is transcribed
        again in a window of its own (DESIGN.md §17.5), so that all of it is heard: the analysis
        only transcribed around listed words (the transcript cache serves what it already heard)."""
        chosen = analysis.subtitles
        cued = [i for i, line in enumerate(lines) if line.cue is not None]
        heard: Sequence[Word] = analysis.heard
        if cued and chosen is not None:
            heard = guided_analysis.transcribe_spans(
                self.guided_context(),
                analysis.strategy,
                audio(),
                [(lines[i].start, lines[i].end) for i in cued],
                chosen.sync.model,
                duration,
            )
            heard = guided_analysis.merge_words(analysis.heard, heard)
        spans = []
        for line in lines:
            words = [
                w
                for w in heard
                if line.start - COVER_MARGIN_S <= (w.start + w.end) / 2 <= line.end + COVER_MARGIN_S
            ]
            if words:
                spans.append((min(w.start for w in words), max(w.end for w in words), True))
            else:
                spans.append((line.start, line.end, False))
        return spans

    def replacer(self, role: str) -> Replacer:
        """Voice replacement (DESIGN.md §16), created once: its models serve every file of a batch. The
        check hears the new words with the transcriber of the analysis (`role`)."""
        if self._replacer is None:
            cfg = self.config
            separator, editor, encoder = self._voice_models or (None, None, None)
            models = VoiceModels(
                cfg.replace,
                device=_context_device(cfg.transcription.device),
                offline=cfg.offline,
                cache_dir=cache_root(cfg) / "voice",
                separator=separator,
                editor=editor,
                encoder=encoder,
            )
            self._replacer = Replacer(
                cfg.replace, self.lexicon, self.ff, models, self._check_transcriber, self.make_room
            )
            self.ui.warn(
                "voice replacement is experimental (DESIGN.md §16): check the review subtitles "
                "(--review-srt); its voice model's weights are licensed for non-commercial use only"
            )
        self._check_role = role
        return self._replacer

    def _check_transcriber(self, audio: Audio, start: float) -> list[Word]:
        _, transcriber = self.transcriber(self._check_role)
        [words] = transcriber.transcribe(
            [Clip(start, audio)], language=self.config.analysis.language, prompt=self.prompt, vad=False
        )
        return list(words)

    def _replace_words(
        self,
        job: Job,
        analysis: Analysis,
        detections: list[Detection],
        to_mute: list[Detection],
        intervals: list[CensorInterval],
        context: ContextResult,
    ) -> None:
        """Say a milder word in place of each listed word the context layer lets through (DESIGN.md §16,
        §17.6). A word's span is replaced only if no other muted word shares it; any failure mutes it."""
        layer = self.context_layer()
        self.make_room("context")  # the judge picks the substitutes
        choices = layer.substitutes(detections, context, self.config.replace.substitutes)
        candidates = []
        for index, (detection, choice) in enumerate(zip(detections, choices, strict=True)):
            verdict = job.verdicts[index] if index < len(job.verdicts) else None
            if verdict is not None:
                verdict["substitute"] = choice.substitute
                verdict["substitute_reason"] = choice.reason
            if choice.substitute is None or detection not in to_mute:
                continue
            middle = (detection.start + detection.end) / 2
            span = next((i for i in intervals if i.start <= middle <= i.end), None)
            if span is None:
                continue
            crowded = any(d != detection and d.start < span.end and span.start < d.end for d in to_mute)
            candidates.append((index, detection, choice.substitute, span, crowded))
        replacer = self.replacer(analysis.strategy)
        planned: list[Candidate | Replacement] = []
        for index, detection, substitute, span, crowded in candidates:
            if crowded:
                planned.append(
                    Replacement(
                        index, span.start, span.end, detection.heard.strip(), substitute, False,
                        "another muted word shares its span",
                    )
                )  # fmt: skip
            else:
                planned.append(
                    replacer.plan(
                        index=index,
                        detection=detection,
                        span=(span.start, span.end),
                        substitute=substitute,
                        heard=analysis.heard,
                        duration=job.info.duration,
                    )
                )
        todo = [c for c in planned if isinstance(c, Candidate)]
        said_again: list[Replacement] = []
        if todo:
            windows = read_pcm_windows(self.ff, job.source, job.stream, [c.window for c in todo])
            with self.ui.progress("Replacing words", 3 * len(todo)) as update:
                said_again = replacer.replace(todo, windows, job.stream, job.workdir, on_progress=update)
        results = iter(said_again)
        done = [next(results) if isinstance(item, Candidate) else item for item in planned]
        job.replacements = done
        if candidates:
            replaced = sum(r.replaced for r in done)
            reasons = sorted({r.reason for r in done if not r.replaced})
            self.ui.info(
                f"Voice replacement: {replaced} of {len(done)} words said again"
                + (f"; the rest stay muted ({'; '.join(reasons)})" if reasons else "")
            )
        job.lap("replace")

    def render_report(
        self,
        source: Path,
        report: Path,
        options: RunOptions,
        *,
        force: bool = False,
        many: bool = False,
        from_folder: bool = False,
    ) -> FileResult:
        """`vbr render`: mute the report's `intervals`, which may have been edited by hand, without
        detecting anything. The report is only read, never rewritten. `from_folder`: as for prepare, a
        file vbr already censored is skipped."""
        cfg = self.config
        started = time.monotonic()
        data = read_report(report)
        intervals = report_intervals(data)
        checked = self._check_output(source, options, many)
        if isinstance(checked, FileResult):
            return checked
        info = probe(self.ff, source)
        tag = info.tags.get(CENSORED_TAG.lower())
        if tag is not None and from_folder:
            return FileResult(source, "skipped", notes=[f"already censored by vbr ({CENSORED_TAG}={tag})"])
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
        items = report_detection_items(data)
        detections = [detection for detection, _ in items]
        said_again = [r for r in data.get("replacements") or [] if isinstance(r, dict) and r.get("replaced")]
        if said_again:
            self.ui.warn(
                f"{len(said_again)} replaced words are muted instead: only vbr clean can replace words"
            )
        subtitle = data.get("subtitle")
        context = data.get("context")
        job = Job(
            source=source,
            options=options,
            many=many,
            info=info,
            stream=stream,
            workdir=Path(tempfile.mkdtemp(prefix="vbr-")),
            result=FileResult(
                source, "scanned", detections=len(detections), intervals=len(intervals), strategy="report"
            ),
            output=checked.output,
            backup=checked.backup,
            started=started,
            clock=started,
            intervals=intervals,
            detections=detections,
            subtitle=subtitle if isinstance(subtitle, dict) else None,
            context=context if isinstance(context, dict) else None,
            verdicts=[v if isinstance(v := item.get("context"), dict) else None for _, item in items],
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
