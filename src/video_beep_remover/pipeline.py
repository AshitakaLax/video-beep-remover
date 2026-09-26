"""Run the stages for one file (DESIGN.md §5.1). Subtitle-guided strategies arrive in milestone M2."""

import contextlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from video_beep_remover import __version__
from video_beep_remover.asr.base import Transcriber, build_prompt
from video_beep_remover.asr.faster_whisper import FasterWhisperTranscriber, ModelChoice, resolve_model
from video_beep_remover.config.loader import LoadedConfig
from video_beep_remover.config.schema import AnalysisConfig
from video_beep_remover.detect.intervals import build_intervals
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.detect.matcher import detect_in_words
from video_beep_remover.errors import ConfigError, UsageError, VbrError
from video_beep_remover.media.audio import SAMPLE_RATE, decode_track
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe, select_audio_stream
from video_beep_remover.media.render import plan_streams, render
from video_beep_remover.report import (
    SCHEMA_VERSION,
    detection_dict,
    edl_text,
    interval_dict,
    write_json,
    write_text,
)

Status = Literal["cleaned", "copied", "clean", "scanned", "skipped"]
Progress = Callable[[float], None]
TranscriberFactory = Callable[[ModelChoice], Transcriber]


class UI(Protocol):
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def progress(self, label: str, total: float) -> AbstractContextManager[Progress]: ...
    def status(self, label: str) -> AbstractContextManager[None]: ...


class NullUI:
    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    @contextlib.contextmanager
    def progress(self, label: str, total: float) -> Iterator[Progress]:
        yield lambda done: None

    @contextlib.contextmanager
    def status(self, label: str) -> Iterator[None]:
        yield


@dataclass(frozen=True)
class RunOptions:
    dry_run: bool = False
    output: Path | None = None  # -o: a file (single input) or a directory
    report: Path | None = None  # --report: a file (single input) or a directory
    edl: bool = False
    overwrite: bool = False
    skip_existing: bool = False
    keep_temp: bool = False


@dataclass
class FileResult:
    input: Path
    status: Status
    output: Path | None = None
    report: Path | None = None
    edl: Path | None = None
    detections: int = 0
    intervals: int = 0
    notes: list[str] = field(default_factory=list)


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


def choose_strategy(analysis: AnalysisConfig) -> tuple[str, str | None]:
    """The strategy that actually runs, and why it differs from the requested one (DESIGN.md §7)."""
    if analysis.strategy == "full":
        return "full", None
    reason = "subtitle-guided analysis is not implemented yet, so no subtitles were searched"
    if not analysis.fallback_to_full:
        raise VbrError(
            f"strategy {analysis.strategy!r} needs subtitles: {reason}, and fallback_to_full is false"
        )
    return "full", reason


def _report_path(options: RunOptions, source: Path, output: Path | None, many: bool) -> Path:
    if options.report is not None:
        if many or options.report.is_dir():
            return options.report / f"{source.stem}.vbr.json"
        return options.report
    base = output or source
    return base.with_name(f"{base.stem}.vbr.json")


class Pipeline:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        ui: UI | None = None,
        ff: FFmpeg | None = None,
        categories: list[str] | None = None,
        transcriber_factory: TranscriberFactory | None = None,
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

    def _load_faster_whisper(self, choice: ModelChoice) -> Transcriber:
        settings = self.config.transcription
        return FasterWhisperTranscriber(
            choice,
            beam_size=settings.beam_size,
            batch_size=settings.batch_size,
            vad_filter=settings.vad_filter,
            offline=self.config.offline,
        )

    def transcriber(self, strategy: str) -> tuple[ModelChoice, Transcriber]:
        choice = resolve_model(
            self.config.transcription, strategy=strategy, language=self.config.analysis.language
        )
        if choice not in self._transcribers:
            with self.ui.status(f"Loading Whisper model {choice.describe()}"):
                self._transcribers[choice] = self._factory(choice)
        return choice, self._transcribers[choice]

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

    def process(self, source: Path, options: RunOptions, *, many: bool = False) -> FileResult:
        cfg = self.config
        timings: dict[str, float] = {}
        started = clock = time.monotonic()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.monotonic()
            timings[name] = round(now - clock, 3)
            clock = now

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
        strategy, fallback_reason = choose_strategy(cfg.analysis)
        self.ui.info(f"{source.name}: {info.duration / 60:.1f} min, analysing audio {stream.describe()}")
        if fallback_reason:
            self.ui.info(f"using the full strategy instead of {cfg.analysis.strategy!r}: {fallback_reason}")
        lap("probe")

        result = FileResult(source, "scanned")
        with self._workdir(options.keep_temp) as workdir:
            with self.ui.progress("Decoding audio", info.duration) as update:
                audio = decode_track(self.ff, source, stream.index, workdir / "audio.f32", on_progress=update)
            lap("decode")
            choice, transcriber = self.transcriber(strategy)
            with self.ui.progress("Transcribing", len(audio) / SAMPLE_RATE) as update:
                words = transcriber.transcribe(
                    audio, offset=0.0, language=cfg.analysis.language, prompt=self.prompt, on_progress=update
                )
            del audio
            lap("transcribe")

            detections = detect_in_words(self.lexicon, words)
            intervals = build_intervals(
                detections,
                duration=info.duration,
                pad_before=cfg.censor.pad_before_ms / 1000,
                pad_after=cfg.censor.pad_after_ms / 1000,
                min_duration=cfg.censor.min_duration_ms / 1000,
                merge_gap=cfg.censor.merge_gap_ms / 1000,
            )
            result.detections, result.intervals = len(detections), len(intervals)
            self.ui.info(f"{len(detections)} listed words found → {len(intervals)} spans to mute")

            report: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "tool_version": __version__,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "input": {"path": str(source), "size": info.size, "duration": round(info.duration, 3)},
                "audio_stream": {
                    "index": stream.index,
                    "codec": stream.codec,
                    "channels": stream.channels,
                    "language": stream.language,
                },
                "strategy": {
                    "requested": cfg.analysis.strategy,
                    "used": strategy,
                    "fallback_reason": fallback_reason,
                },
                "transcription": {
                    "model": choice.name,
                    "device": choice.device,
                    "compute_type": choice.compute_type,
                    "prompt": self.prompt,
                    "words": len(words),
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
