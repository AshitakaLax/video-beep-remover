"""OpenSubtitles.com as a subtitle source (DESIGN.md §6.3-6.4).

Subtitles downloaded for this video before are offered first, from the cache, even offline or
without an API key. Online, the movie hash is searched first; subtitles that match it were timed
against this exact file, so they are trusted. Only when nothing matches the hash is the title
(or an IMDb id from a .nfo file) searched. Each download counts against the user's daily quota, so
only the candidates that are actually tried are downloaded, and every download is cached."""

import math
from pathlib import Path
from typing import Any

from video_beep_remover.config.schema import Config
from video_beep_remover.errors import SubtitleError
from video_beep_remover.media.probe import MediaInfo
from video_beep_remover.subtitles.acquire import SubtitleCandidate
from video_beep_remover.subtitles.cache import CachedSubtitle, SubtitleCache
from video_beep_remover.subtitles.names import guess, imdb_id_from_nfo, release_similarity
from video_beep_remover.subtitles.opensubtitles import (
    OnlineSubtitle,
    OpenSubtitlesClient,
    OpenSubtitlesError,
    QuotaExceeded,
)
from video_beep_remover.subtitles.oshash import opensubtitles_hash
from video_beep_remover.subtitles.parse import read_subtitle_file

HASH_MATCH_BONUS = 40.0  # timed against this exact file
FPS_MATCH_BONUS = 5.0
MAX_POPULARITY_BONUS = 5.0  # log10 of the download count, as a tie-breaker


class OnlineSubtitles:
    def __init__(
        self, config: Config, info: MediaInfo, *, client: OpenSubtitlesClient | None, cache: SubtitleCache
    ) -> None:
        self.settings = config.subtitles.opensubtitles
        self.languages = list(config.subtitles.languages)
        self.offline = config.offline
        self.info = info
        self.client = client
        self.cache = cache
        self.movie_hash = opensubtitles_hash(info.path)
        self.fingerprint = f"{self.movie_hash}-{info.path.stat().st_size}" if self.movie_hash else None
        self.video = guess(info.path.name)
        self.quota: QuotaExceeded | None = None
        self.report: dict[str, Any] = {"movie_hash": self.movie_hash, "searches": [], "downloads": 0}

    def planned_searches(self) -> list[dict[str, Any]]:
        """What would be sent: the movie hash, then the title (DESIGN.md §10)."""
        searches: list[dict[str, Any]] = []
        if self.movie_hash:
            searches.append({"moviehash": self.movie_hash})
        imdb_id = imdb_id_from_nfo(self.info.path) if self.video.kind == "movie" else None
        if imdb_id is not None:
            searches.append({"imdb_id": imdb_id})
        elif self.video.title:
            by_title: dict[str, Any] = {"query": self.video.title}
            if (
                self.video.kind == "episode"
                and self.video.season is not None
                and self.video.episode is not None
            ):
                by_title |= {"season": self.video.season, "episode": self.video.episode}
            elif self.video.year:
                by_title["year"] = self.video.year
            searches.append(by_title)
        return searches

    def _candidate(self, found: OnlineSubtitle | CachedSubtitle, *, cached: bool) -> SubtitleCandidate:
        release = found.release
        bonus = HASH_MATCH_BONUS if found.moviehash_match else 0.0
        if release:
            bonus += release_similarity(self.video, guess(release))
        if found.fps and self.info.frame_rate and abs(found.fps - self.info.frame_rate) < 0.01:
            bonus += FPS_MATCH_BONUS
        bonus += min(MAX_POPULARITY_BONUS, math.log10(1 + found.download_count))
        name = release or (found.file_name if isinstance(found, OnlineSubtitle) else None) or ""
        label = f"OpenSubtitles #{found.file_id}" + (f" {name}" if name else "")
        label += " (hash match)" if found.moviehash_match else ""
        return SubtitleCandidate(
            source="opensubtitles",
            label=label,
            language=(found.language or "").lower() or None,
            hearing_impaired=found.hearing_impaired,
            forced=isinstance(found, OnlineSubtitle) and found.foreign_parts_only,
            trusted=found.moviehash_match,
            file_id=found.file_id,
            release=release,
            fps=found.fps,
            bonus=bonus,
            cached=cached,
        )

    def find(self) -> tuple[list[SubtitleCandidate], list[str]]:
        """Candidates (unranked) and notes on what could not be searched."""
        if not self.settings.enabled:
            return [], ["OpenSubtitles: disabled (subtitles.opensubtitles.enabled = false)"]
        known = (
            {entry.file_id: entry for entry in self.cache.known(self.fingerprint)} if self.fingerprint else {}
        )
        notes: list[str] = []
        found: dict[int, OnlineSubtitle] = {}
        if self.offline:
            notes.append(f"OpenSubtitles: offline; {len(known)} subtitles cached for this file")
        elif self.client is None:
            notes.append(
                "OpenSubtitles: skipped, no API key (set OPENSUBTITLES_API_KEY to your own free key)"
                + (f"; {len(known)} subtitles cached for this file" if known else "")
            )
        elif not (searches := self.planned_searches()):
            notes.append("OpenSubtitles: nothing to search by (no movie hash, and no title in the file name)")
        else:
            try:
                for params in searches:
                    if "moviehash" not in params and any(r.moviehash_match for r in found.values()):
                        break  # subtitles timed for this very file beat anything a title search finds
                    results = self.client.search(languages=self.languages, **params)
                    self.report["searches"].append(params | {"results": len(results)})
                    for result in results:
                        found.setdefault(result.file_id, result)
            except OpenSubtitlesError as exc:
                notes.append(f"OpenSubtitles: {exc}")
        exclude = self.settings.exclude_machine_translated
        candidates = [
            self._candidate(result, cached=self.cache.path(file_id) is not None)
            for file_id, result in found.items()
            if not (exclude and result.machine_translated)
        ]
        candidates += [
            self._candidate(entry, cached=True) for file_id, entry in known.items() if file_id not in found
        ]
        return candidates, notes

    def fetch(self, candidate: SubtitleCandidate) -> str:
        """The subtitle text: from the cache, or downloaded (which counts against the quota)."""
        if candidate.file_id is None:
            raise SubtitleError(f"{candidate.label}: no file to fetch")
        path = self.cache.path(candidate.file_id)
        if path is not None:
            text = read_subtitle_file(path, candidate.language)
            self._remember(candidate, path)
            return text
        if self.offline or self.client is None:
            raise SubtitleError("not cached, and OpenSubtitles cannot be reached")
        if self.quota is not None:
            raise SubtitleError(f"not downloaded: {self.quota}")
        try:
            download = self.client.download(candidate.file_id)
        except QuotaExceeded as exc:
            self.quota = exc
            self.report["quota"] = {"remaining": 0, "reset_time_utc": exc.reset_time_utc}
            raise SubtitleError(str(exc)) from exc
        except OpenSubtitlesError as exc:
            raise SubtitleError(str(exc)) from exc
        self.report["downloads"] += 1
        self.report["quota"] = {"remaining": download.remaining, "reset_time_utc": download.reset_time_utc}
        suffix = Path(download.file_name).suffix if download.file_name else ".srt"
        path = self.cache.store(candidate.file_id, download.data, suffix or ".srt")
        self._remember(candidate, path)
        return read_subtitle_file(path, candidate.language)

    def _remember(self, candidate: SubtitleCandidate, path: Path) -> None:
        if self.fingerprint is None or candidate.file_id is None:
            return
        self.cache.remember(
            self.fingerprint,
            CachedSubtitle(
                file_id=candidate.file_id,
                file=path.name,
                language=candidate.language,
                hearing_impaired=candidate.hearing_impaired,
                fps=candidate.fps,
                release=candidate.release,
                moviehash_match=candidate.trusted,
                download_count=0,
            ),
        )
