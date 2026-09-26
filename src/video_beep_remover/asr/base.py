"""Speech-recognition interface (DESIGN.md §5.4)."""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.models import Word

ProgressCallback = Callable[[float], None]
PROMPT_WORDS = 8


class Transcriber(Protocol):
    name: str

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        *,
        offset: float,
        language: str,
        prompt: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> Sequence[Word]:
        """Words with times in media seconds (`offset` is the media time of sample 0).

        `on_progress` receives how many seconds of `audio` have been transcribed.
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
