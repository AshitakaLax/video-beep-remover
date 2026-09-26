"""Run the stages for one file (DESIGN.md §5.1) and choose between strategies (§7)."""

import contextlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from video_beep_remover import __version__
from video_beep_remover import guided as guided_analysis
from video_beep_remover.asr.base import Clip, Transcriber, build_prompt
from video_beep_remover.asr.faster_whisper import (
    FasterWhisperTranscriber,
    ModelChoice,
    resolve_anchor_model,
    resolve_model,
)
from video_beep_remover.asr.vad import SpeechDetector, silero_speech
from video_beep_remover.config.loader import LoadedConfig, cache_root
from video_beep_remover.detect.intervals import build_intervals
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.detect.matcher import detect_in_words
from video_beep_remover.errors import ConfigError, UsageError, VbrError
from video_beep_remover.media.audio import (
    SAMPLE_RATE,
    ArrayAudioSource,
    Audio,
    AudioSource,
    SeekingAudioSource,
    decode_track,
)
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import MediaInfo, StreamInfo, probe, select_audio_stream
from video_beep_remover.media.render import plan_streams, render
from video_beep_remover.models import Detection
from video_beep_remover.report import (
    SCHEMA_VERSION,
    detection_dict,
    edl_text,
    interval_dict,
    write_json,
    write_text,
)
from video_beep_remover.subtitles.cache import SubtitleCache
from video_beep_remover.subtitles.online import OnlineSubtitles
from video_beep_remover.subtitles.opensubtitles import OpenSubtitlesClient
from video_beep_remover.subtitles.oshash import opensubtitles_hash
from video_beep_remover.ui import UI, NullUI, Progress

__all__ = ["UI", "FileResult", "NullUI", "Pipeline", "Progress", "RunOptions", "resolve_output"]

Status = Literal["cleaned", "copied", "clean", "scanned", "skipped"]
TranscriberFactory = Callable[[ModelChoice], Transcriber]


@dataclass(frozen=True)
class RunOptions:
    dry_run: bool = False
    output: Path | None = None  # -o: a file (single input) or a directory
    report: Path | None = None  # --report: a file (single input) or a directory
    edl: bool = False
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
    detections: int = 0
    intervals: int = 0
    strategy: str = ""  # the strategy that ran
    notes: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    strategy: str  # the one that ran
    fallback_reason: str | None
    detections: list[Detection]
    words: int
    model: ModelChoice | None
    report: dict[str, Any]  # strategy-specific report sections


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
        self._factory = transcriber_factory or self._load_faster_whisper
        self._transcribers: dict[ModelChoice, Transcriber] = {}
        self.detect_speech: SpeechDetector = speech_detector or silero_speech
        self.subtitle_cache = SubtitleCache(cache_root(self.config))
        self._opensubtitles = opensubtitles

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

    def _load_faster_whisper(self, choice: ModelChoice) -> Transcriber:
        settings = self.config.transcription
        return FasterWhisperTranscriber(
            choice,
            beam_size=settings.beam_size,
            batch_size=settings.batch_size,
            vad_filter=settings.vad_filter,
            offline=self.config.offline,
        )

    def transcriber(self, role: str) -> tuple[ModelChoice, Transcriber]:
        """The model for a role: "full", "targeted" or "hybrid" (by strategy), or "anchor"."""
        settings, language = self.config.transcription, self.config.analysis.language
        if role == "anchor":
            choice = resolve_anchor_model(settings, language=language)
        else:
            choice = resolve_model(settings, strategy=role, language=language)
        if choice not in self._transcribers:
            with self.ui.status(f"Loading Whisper model {choice.describe()}"):
                self._transcribers[choice] = self._factory(choice)
        return choice, self._transcribers[choice]

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
        )

    @contextlib.contextmanager
    def _workdir(self, keep: bool) -> Iterator[Path]:
        path = Path(tempfile.mkdtemp(prefix="vbr-"))
        try:
            yield path
        finally:
            if keep:
                self.ui.info(f"kept temporary files in {path}")
            else:
                shutil.rmtree(path, ignore_errors=True)

    def _full(
        self, track: _Track, lap: Callable[[str], None], fallback_reason: str | None, report: dict[str, Any]
    ) -> Analysis:
        audio = track.get()
        lap("decode")
        choice, transcriber = self.transcriber("full")
        with self.ui.progress("Transcribing", len(audio) / SAMPLE_RATE) as update:
            [words] = transcriber.transcribe(
                [Clip(0.0, audio)],
                language=self.config.analysis.language,
                prompt=self.prompt,
                vad=self.config.transcription.vad_filter,
                on_progress=update,
            )
        lap("transcribe")
        return Analysis(
            "full", fallback_reason, detect_in_words(self.lexicon, words), len(words), choice, report
        )

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
        audio: AudioSource
        if requested == "hybrid":
            audio = ArrayAudioSource(track.get())  # hybrid needs the whole track for VAD anyway
            lap("decode")
        else:
            audio = SeekingAudioSource(self.ff, source, stream.index)
        try:
            found = guided_analysis.analyse(
                self.guided_context(),
                requested,
                info,
                audio,
                stream.index,
                workdir,
                explicit=subtitles,
                track=track.audio,
                lap=lap,
            )
        except guided_analysis.Fallback as fallback:
            if not analysis.fallback_to_full:
                raise VbrError(
                    f"strategy {requested!r} could not run: {fallback.reason}; fallback_to_full is false"
                ) from fallback
            self.ui.info(f"using the full strategy instead of {requested!r}: {fallback.reason}")
            return self._full(track, lap, fallback.reason, fallback.report)
        return Analysis(requested, None, found.detections, found.words, found.model, found.report)

    def process(self, source: Path, options: RunOptions, *, many: bool = False) -> FileResult:
        cfg = self.config
        timings: dict[str, float] = {}
        started = clock = time.monotonic()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.monotonic()
            timings[name] = round(timings.get(name, 0.0) + now - clock, 3)
            clock = now

        if options.subtitles is not None and not options.subtitles.is_file():
            raise UsageError(f"subtitle file not found: {options.subtitles}")
        output = None
        if not options.dry_run:
            output = resolve_output(source, cfg.output.path, options.output, many=many)
            if output.resolve() == source.resolve():
                raise UsageError(f"the output would overwrite the input: {output}")
            if output.exists() and not (options.overwrite or cfg.output.overwrite):
                if options.skip_existing:
                    return FileResult(source, "skipped", output=output, notes=["output already exists"])
                raise UsageError(f"{output} already exists (use --overwrite or --skip-existing)")

        info = probe(self.ff, source)
        stream = select_audio_stream(info, cfg.analysis.language, cfg.analysis.audio_stream)
        self.ui.info(f"{source.name}: {info.duration / 60:.1f} min, analysing audio {stream.describe()}")
        lap("probe")

        result = FileResult(source, "scanned")
        with self._workdir(options.keep_temp) as workdir:
            track = _Track(self, source, info, stream, workdir)
            try:
                analysis = self._analyse(source, info, stream, workdir, track, lap, options.subtitles)
            finally:
                track.release()
            detections = analysis.detections
            intervals = build_intervals(
                detections,
                duration=info.duration,
                pad_before=cfg.censor.pad_before_ms / 1000,
                pad_after=cfg.censor.pad_after_ms / 1000,
                min_duration=cfg.censor.min_duration_ms / 1000,
                merge_gap=cfg.censor.merge_gap_ms / 1000,
            )
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
            report: dict[str, Any] = {
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
                    "model": model.name if model else None,
                    "device": model.device if model else None,
                    "compute_type": model.compute_type if model else None,
                    "prompt": self.prompt,
                    "words": analysis.words,
                },
                "categories": list(self.lexicon.categories),
                "detections": [detection_dict(d) for d in detections],
                "intervals": [interval_dict(i) for i in intervals],
                "output": None,
                "timings": timings,
            }

            if options.dry_run:
                result.status = "scanned"
            elif not intervals and cfg.output.when_clean == "skip":
                result.status = "clean"
            else:
                assert output is not None
                plan = plan_streams(info, stream, cfg.output, cfg.analysis.language)
                for note in plan.notes:
                    self.ui.warn(note)
                output.parent.mkdir(parents=True, exist_ok=True)
                with self.ui.progress("Rendering", info.duration) as update:
                    rendered = render(
                        self.ff,
                        info,
                        plan,
                        intervals,
                        output=output,
                        fade=cfg.censor.fade_ms / 1000,
                        output_config=cfg.output,
                        workdir=workdir,
                        on_progress=update,
                    )
                lap("render")
                result.status = "cleaned" if intervals else "copied"
                result.output = output
                result.notes = list(plan.notes)
                report["output"] = {
                    "path": str(output),
                    "encoders": {str(index): encoder for index, encoder in rendered.encoders.items()},
                    "muted_spans": [interval_dict(i) for i in rendered.intervals],
                    "verified_spans": rendered.verified_spans,
                    "timeline_shift": round(rendered.timeline_shift, 3),
                    "notes": list(plan.notes),
                }

        timings["total"] = round(time.monotonic() - started, 3)
        if options.report is not None or cfg.output.report:
            result.report = _report_path(options, source, output, many)
            write_json(result.report, report)
        if options.edl or cfg.output.edl:
            edl = source.with_suffix(".edl")
            if edl.exists() and not (options.overwrite or cfg.output.overwrite):
                self.ui.warn(f"not overwriting existing {edl.name} (use --overwrite)")
            else:
                write_text(edl, edl_text(intervals))
                result.edl = edl
        return result
