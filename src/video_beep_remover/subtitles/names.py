"""What a file name says about a video: title, year, episode and release (DESIGN.md §6.3-6.4)."""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

_IMDB_ID = re.compile(r"\btt(\d{7,9})\b")
_NFO_BYTES = 256 * 1024
# How much agreeing on each release field says about matching timing. A different edition or
# release group usually means a different cut or different timing; resolution and codec say little.
RELEASE_WEIGHTS = {
    "edition": 10,
    "release_group": 10,
    "source": 5,
    "streaming_service": 5,
    "screen_size": 1,
    "video_codec": 1,
}


@dataclass(frozen=True)
class MediaGuess:
    kind: Literal["movie", "episode"]
    title: str | None = None
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    release: dict[str, str] = field(default_factory=dict)  # RELEASE_WEIGHTS fields, lower-cased


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _int(value: Any) -> int | None:
    value = _first(value)
    return value if isinstance(value, int) else None


@lru_cache(maxsize=512)
def guess(name: str) -> MediaGuess:
    """Parse a file or release name with guessit."""
    from guessit import guessit

    found = dict(guessit(name))
    title = _first(found.get("title"))
    release = {
        key: str(_first(found[key])).lower()
        for key in RELEASE_WEIGHTS
        if found.get(key) not in (None, "", [])
    }
    return MediaGuess(
        kind="episode" if found.get("type") == "episode" else "movie",
        title=str(title) if title else None,
        year=_int(found.get("year")),
        season=_int(found.get("season")),
        episode=_int(found.get("episode")),
        release=release,
    )


def release_similarity(video: MediaGuess, release: MediaGuess) -> int:
    """Weighted count of release fields (group, source, edition...) the two names agree on."""
    return sum(
        weight
        for key, weight in RELEASE_WEIGHTS.items()
        if key in video.release and video.release[key] == release.release.get(key)
    )


def imdb_id_from_nfo(video: Path) -> int | None:
    """The IMDb id in a Kodi-style `<stem>.nfo` or `movie.nfo` next to the video, as a number."""
    for nfo in (video.with_suffix(".nfo"), video.parent / "movie.nfo"):
        try:
            with nfo.open("rb") as file:
                text = file.read(_NFO_BYTES).decode("utf-8", "replace")
        except OSError:
            continue
        match = _IMDB_ID.search(text)
        if match:
            return int(match.group(1))
    return None
