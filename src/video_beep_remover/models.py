"""Data types shared by all stages. Times are seconds on the media timeline (DESIGN.md §6.2)."""

from dataclasses import dataclass
from typing import Literal

DetectionSource = Literal["asr", "estimate", "cue"]


@dataclass(frozen=True, slots=True)
class Word:
    """One recognized word."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class Detection:
    """A listed word heard (or estimated) in the audio."""

    start: float
    end: float
    heard: str
    term: str
    category: str
    confidence: float
    source: DetectionSource = "asr"
    cue: int | None = None


@dataclass(frozen=True, slots=True)
class CensorInterval:
    """A span the renderer mutes. Lists of intervals are always sorted and disjoint."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Cue:
    """One cleaned subtitle cue. Times are subtitle time until a SyncModel maps them."""

    index: int  # 1-based, in time order after cleaning
    start: float
    end: float
    text: str  # what is spoken: markup, speaker labels and sound descriptions removed
    lyrics: bool = False


@dataclass(frozen=True, slots=True)
class SyncModel:
    """Maps subtitle time to media time: t_media = scale * t_sub + offset."""

    scale: float = 1.0
    offset: float = 0.0
    error: float = 0.0  # median absolute residual of the anchors, in seconds

    def to_media(self, t: float) -> float:
        return self.scale * t + self.offset


@dataclass(frozen=True, slots=True)
class Window:
    """A stretch of audio to transcribe, and why."""

    start: float
    end: float
    reasons: frozenset[str] = frozenset()  # "lexicon", "masked", "hint", "uncovered", "expanded"
    cues: tuple[int, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start
