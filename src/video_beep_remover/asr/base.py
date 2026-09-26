"""Speech-recognition interface (DESIGN.md §5.4)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.media.audio import SAMPLE_RATE, Audio
from video_beep_remover.models import Word

ProgressCallback = Callable[[float], None]
PROMPT_WORDS = 8


@dataclass(frozen=True)
class Clip:
    """Audio to transcribe: 16 kHz mono float32 whose sample 0 is `start` on the media timeline."""

    start: float
    audio: Audio

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class Transcriber(Protocol):
    name: str

    def transcribe(
        self,
        clips: Sequence[Clip],
        *,
        language: str,
        prompt: str | None,
        vad: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[Word]]:
        """The words heard in each clip, with times in media seconds.

        `vad` skips non-speech inside long clips (full mode). `on_progress` receives how many
        seconds of audio have been transcribed so far, over all clips.
        """
        ...


def build_prompt(setting: str, lexicon: Lexicon) -> str | None:
    """The initial prompt: "" turns it off, "auto" lists a few enabled words verbatim, anything
    else is used as written. An uncensored prompt nudges Whisper toward uncensored spelling."""
    if not setting.strip():
        return None
    if setting != "auto":
        return setting
    words: list[str] = []
    for term in lexicon.terms:
        if len(term.words) != 1 or term.words[0].search:
            continue
        core = term.words[0].text.replace("*", "")
        if core.isalpha() and core not in words:
            words.append(core)
        if len(words) == PROMPT_WORDS:
            break
    return (", ".join(words).capitalize() + ".") if words else None
