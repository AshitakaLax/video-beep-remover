"""Find speech in a decoded track with the Silero VAD model that ships with faster-whisper.

`hybrid` uses it to find speech that no subtitle cue covers (DESIGN.md §6.7 step 4)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from video_beep_remover.asr.base import Clip
from video_beep_remover.errors import DependencyError
from video_beep_remover.media.audio import SAMPLE_RATE, Audio
from video_beep_remover.models import Word

CHUNK_S = 600  # the model runs on 10-minute chunks, which keeps memory flat on long films
JOIN_S = 0.5  # speech regions this close together (e.g. across a chunk border) are joined

Regions = list[tuple[float, float]]


class SpeechDetector(Protocol):
    def __call__(self, audio: Audio, on_progress: Callable[[float], None] | None = None) -> Regions:
        """Speech regions in media seconds; `audio` sample 0 is the start of the file."""
        ...


def silero_speech(audio: Audio, on_progress: Callable[[float], None] | None = None) -> Regions:
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:
        raise DependencyError("faster-whisper is not installed: pip install faster-whisper") from exc
    options = VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=500, speech_pad_ms=200)
    regions: Regions = []
    step = CHUNK_S * SAMPLE_RATE
    for offset in range(0, len(audio), step):
        chunk = np.ascontiguousarray(audio[offset : offset + step], dtype=np.float32)
        for found in get_speech_timestamps(chunk, options, sampling_rate=SAMPLE_RATE):
            start = (offset + found["start"]) / SAMPLE_RATE
            end = (offset + found["end"]) / SAMPLE_RATE
            if regions and start - regions[-1][1] < JOIN_S:
                regions[-1] = (regions[-1][0], max(regions[-1][1], end))
            else:
                regions.append((start, end))
        if on_progress is not None:
            on_progress(min(len(audio), offset + step) / SAMPLE_RATE)
    return regions


TRIM_PAD_S = 0.1  # extra non-speech kept around the detected speech, which VAD already pads by 0.2 s
MIN_TRIM_S = 0.1  # trimming less than this is not worth it
MIN_SHIFT_S = 0.3  # snap_to_speech moves a word only this much or more


@dataclass(frozen=True)
class Trimmed:
    clip: Clip  # without its leading and trailing non-speech
    clean_start: bool  # trimmed at the start: no word can be cut there
    clean_end: bool
    speech: Regions  # the speech found, in media time


def trim_to_speech(clip: Clip, detect: SpeechDetector) -> Trimmed:
    """The clip without its leading and trailing non-speech, and whether its start and end were trimmed.

    After a pause, Whisper tends to start the first word at the very beginning of the clip, up to
    seconds early. Trimming the pause keeps word times honest; the trimmed edges fall in silence, so
    no word is cut there. A clip without detected speech is returned whole, since VAD can miss
    shouting or singing."""
    regions = detect(clip.audio)
    speech = [(clip.start + start, clip.start + end) for start, end in regions]
    if not regions:
        return Trimmed(clip, False, False, speech)
    head = max(0.0, regions[0][0] - TRIM_PAD_S)
    tail = min(clip.duration, regions[-1][1] + TRIM_PAD_S)
    trim_head = head >= MIN_TRIM_S
    trim_tail = clip.duration - tail >= MIN_TRIM_S
    first = round(head * SAMPLE_RATE) if trim_head else 0
    last = round(tail * SAMPLE_RATE) if trim_tail else len(clip.audio)
    trimmed = Clip(clip.start + first / SAMPLE_RATE, clip.audio[first:last])
    return Trimmed(trimmed, trim_head, trim_tail, speech)


def snap_to_speech(words: Sequence[Word], speech: Regions) -> list[Word]:
    """Words that start in a pause, moved to where the speech resumes.

    The same Whisper habit as above, inside a clip: the first word after a pause often starts where
    the previous speech ended, seconds early. A clip that holds several lines, like a sync anchor's,
    keeps those pauses after trimming. A word that starts at least MIN_SHIFT_S before the next speech
    region and ends inside it starts at that region instead."""
    snapped = []
    for word in words:
        if not any(start <= word.start <= end for start, end in speech):
            resume = next((start for start, _ in speech if start > word.start), None)
            if resume is not None and resume - word.start >= MIN_SHIFT_S and word.end > resume:
                word = Word(word.text, resume, word.end, word.probability)
        snapped.append(word)
    return snapped
