"""What a file name says about a video: title, year, episode and release (DESIGN.md §6.3-6.4), and what
the Kodi-style .nfo files next to it add."""

import dataclasses
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

_IMDB_ID = re.compile(r"\btt(\d{7,9})\b")
_NFO_BYTES = 256 * 1024
_EPISODE_FIRST = re.compile(r"^\W*(s\d{1,4}\W*e\d{1,4}|\d{1,2}x\d{1,3})\b", re.IGNORECASE)  # "S01E02 - Pilot"
_SEASON_FOLDER = re.compile(r"^((season|series|staffel|saison|temporada)\W*\d{1,4}|s\d{1,4}|specials)$", re.I)
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


def _nfo(path: Path, root: str) -> ElementTree.Element | None:
    """A Kodi-style .nfo file, when it holds XML whose root element is `root`."""
    try:
        with path.open("rb") as file:
            element = ElementTree.fromstring(file.read(_NFO_BYTES))
    except (OSError, ElementTree.ParseError):
        return None
    return element if element.tag == root else None


def _text(element: ElementTree.Element, tag: str) -> str | None:
    value = (element.findtext(tag) or "").strip()
    return value or None


@dataclass(frozen=True)
class EpisodeNfo:
    show: str | None  # <showtitle>
    imdb_id: int | None  # the episode's own


def episode_nfo(video: Path) -> EpisodeNfo | None:
    """What the episode's own `<stem>.nfo` (<episodedetails>, as Kodi, Jellyfin or Emby write it) says."""
    found = _nfo(video.with_suffix(".nfo"), "episodedetails")
    if found is None:
        return None
    ids = [u.text or "" for u in found.iter("uniqueid") if u.get("type") == "imdb"]
    ids.append(found.findtext("imdbid") or "")
    imdb_id = next((int(match.group(1)) for text in ids if (match := _IMDB_ID.search(text))), None)
    return EpisodeNfo(_text(found, "showtitle"), imdb_id)


def _show_from_folders(video: Path) -> str | None:
    """The show of an episode in a library: the title in the show folder's tvshow.nfo, or the name of
    the folder that holds the "Season 01" folder ("MythBusters (2003)/Season 01/S01E02.mkv"). Any
    other folder name says nothing: it could be "Downloads"."""
    in_season = _SEASON_FOLDER.match(video.parent.name) is not None
    folder = video.parent.parent if in_season else video.parent
    nfo = _nfo(folder / "tvshow.nfo", "tvshow")
    if nfo is not None and (title := _text(nfo, "title")):
        return title
    return guess(folder.name).title if in_season else None


def guess_video(video: Path) -> MediaGuess:
    """guess() of the file name; for an episode, the show is the one its .nfo names. guessit takes the
    episode's title for the show's when the name starts with the episode ("S01E02 - Pilot.mkv"): the
    show then comes from the library's folders, or stays unknown, and no title is searched."""
    found = guess(video.name)
    if found.kind != "episode":
        return found
    nfo = episode_nfo(video)
    show = nfo.show if nfo else None
    if show is None and _EPISODE_FIRST.match(video.stem):
        show = _show_from_folders(video)
        if show is None:
            return dataclasses.replace(found, title=None)
    return dataclasses.replace(found, title=show) if show else found


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
