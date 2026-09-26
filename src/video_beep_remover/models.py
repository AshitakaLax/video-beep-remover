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
