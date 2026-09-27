"""Speech recognition (DESIGN.md §6.8): the Transcriber protocol, the faster-whisper and WhisperX
backends, speech detection and the transcript cache."""

from video_beep_remover.asr.base import Transcriber, build_prompt
from video_beep_remover.asr.faster_whisper import FasterWhisperTranscriber, ModelChoice, resolve_model

__all__ = ["FasterWhisperTranscriber", "ModelChoice", "Transcriber", "build_prompt", "resolve_model"]
