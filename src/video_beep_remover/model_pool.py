"""The large models of a run: Whisper by role, the context layer (DESIGN.md §17) and voice replacement
(§16). Each loads on first use and serves every file of a batch. Before one runs, the others are dropped
(models.keep_loaded = false), so only one of them is in memory at a time."""

import gc
import logging
import sys
from collections.abc import Callable
from typing import Any

import httpx

from video_beep_remover.asr.base import Clip, Transcriber
from video_beep_remover.asr.cuda import load_pip_libraries
from video_beep_remover.asr.faster_whisper import (
    FasterWhisperTranscriber,
    ModelChoice,
    cuda_available,
    resolve_anchor_model,
    resolve_model,
)
from video_beep_remover.config.loader import cache_root
from video_beep_remover.config.schema import Config
from video_beep_remover.context import ContextLayer, ModelFactory, torch_device
from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.media.audio import Audio
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.models import Word
from video_beep_remover.ui import UI
from video_beep_remover.voice import Replacer, VoiceModels

log = logging.getLogger(__name__)

TranscriberFactory = Callable[[ModelChoice], Transcriber]
VoiceFactories = tuple[Callable[[], Any], Callable[[], Any], Callable[[], Any]]


def _free_memory() -> None:
    """Return the memory of dropped models: Python's, and the GPU memory PyTorch keeps cached."""
    gc.collect()
    torch = sys.modules.get("torch")  # only if a model already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelPool:
    def __init__(
        self,
        config: Config,
        ui: UI,
        lexicon: Lexicon,
        prompt: str | None,
        ff: FFmpeg,
        *,
        transcriber_factory: TranscriberFactory | None = None,
        context_models: tuple[ModelFactory, ModelFactory] | None = None,
        voice_models: VoiceFactories | None = None,
    ) -> None:
        """The factories stand in for the models in tests: (classifier, judge) and (separator, editor,
        speaker encoder)."""
        if config.transcription.device != "cpu":
            # Before anything imports ctranslate2, even the speech detector (asr/cuda.py).
            load_pip_libraries()
        self.config = config
        self.ui = ui
        self.lexicon = lexicon
        self.prompt = prompt
        self.ff = ff
        self.context_models = context_models
        self.voice_models = voice_models
        self._factory = transcriber_factory or self._load_transcriber
        self._transcribers: dict[ModelChoice, Transcriber] = {}
        self._context: ContextLayer | None = None
        self._replacer: Replacer | None = None
        self._check_role = "full"  # the transcriber that checks replaced words: the analysis's
        self._device: str | None = None  # of the context and voice models, once decided

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

    def hear(self, audio: Audio, start: float) -> list[Word]:
        """The words in 16 kHz mono audio that starts at a media time, heard by the analysis's model: how
        replaced words are checked (voice/, media/render.py)."""
        _, transcriber = self.transcriber(self._check_role)
        [words] = transcriber.transcribe(
            [Clip(start, audio)], language=self.config.analysis.language, prompt=self.prompt, vad=False
        )
        return list(words)

    def device(self) -> str:
        """Where the context and voice models run (context.models.torch_device), decided once, with a
        warning when PyTorch cannot use the GPU that Whisper uses."""
        if self._device is None:
            setting = self.config.transcription.device
            self._device = torch_device(setting)
            if setting == "auto" and self._device == "cpu" and cuda_available():
                self.ui.warn(
                    "PyTorch cannot use the GPU that Whisper uses (is it a CPU-only build?), so the "
                    "context and voice models run on the CPU"
                )
        return self._device

    def context_layer(self) -> ContextLayer:
        """The context layer (DESIGN.md §17), created once: its models serve every file of a batch."""
        if self._context is None:
            classifier, judge = self.context_models or (None, None)
            self._context = ContextLayer(
                self.config,
                device=self.device(),
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

    def replacer(self, role: str) -> Replacer:
        """Voice replacement (DESIGN.md §16), created once: its models serve every file of a batch. The
        check hears the new words with the transcriber of the analysis (`role`)."""
        if self._replacer is None:
            cfg = self.config
            separator, editor, encoder = self.voice_models or (None, None, None)
            models = VoiceModels(
                cfg.replace,
                device=self.device(),
                offline=cfg.offline,
                cache_dir=cache_root(cfg) / "voice",
                separator=separator,
                editor=editor,
                encoder=encoder,
            )
            self._replacer = Replacer(cfg.replace, self.lexicon, self.ff, models, self.hear, self.make_room)
            self.ui.warn(
                "voice replacement is experimental (DESIGN.md §16): check the review subtitles "
                "(--review-srt); its voice model's weights are licensed for non-commercial use only"
            )
        self._check_role = role
        return self._replacer
