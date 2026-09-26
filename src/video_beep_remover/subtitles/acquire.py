"""Find subtitle candidates for a video, rank them and load their text (DESIGN.md §6.3)."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from video_beep_remover.config.schema import SubtitlesConfig
from video_beep_remover.errors import MediaError, SubtitleError
from video_beep_remover.languages import lang_matches, language_code
from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import VIDEO_SUFFIXES, MediaInfo
from video_beep_remover.subtitles.parse import read_subtitle_file

CandidateSource = Literal["explicit", "embedded", "sidecar", "opensubtitles"]

SIDECAR_SUFFIXES = frozenset({".srt", ".ass", ".ssa", ".vtt"})
SUBTITLE_DIRS = frozenset({"subs", "subtitles"})
_SDH_TITLE = re.compile(r"\bSDH\b|\bCC\b|hearing|closed.?caption", re.IGNORECASE)
_FORCED_TITLE = re.compile(r"forced|foreign|signs", re.IGNORECASE)
_COMMENTARY_TITLE = re.compile(r"comment", re.IGNORECASE)
_NAME_TOKEN = re.compile(r"[^\s._\-\[\]()]+")
_SDH_TOKENS = frozenset({"sdh", "cc", "hearing", "impaired"})
_FORCED_TOKENS = frozenset({"forced", "foreign"})


@dataclass(frozen=True)
class SubtitleCandidate:
    source: CandidateSource
    label: str  # for messages and the report, e.g. 'embedded #4 "English SDH"' or "Subs/English.srt"
    language: str | None  # ISO 639-1 code, None when unknown
    hearing_impaired: bool = False
    forced: bool = False
    trusted: bool = True  # timed for this very file, so the sync check searches only a few seconds
    path: Path | None = None  # sidecar or explicit file
    stream: int | None = None  # embedded stream index
    codec: str | None = None
    bonus: float = 0.0  # source-specific preference, e.g. the default flag on an embedded stream
    score: float = 0.0
    file_id: int | None = None  # OpenSubtitles file
    release: str | None = None  # the release the subtitles were made for, e.g. "Movie.2019.1080p.BluRay-GRP"
    fps: float | None = None  # the frame rate they were made for, when known
    cached: bool = False  # downloaded before: trying it costs no download quota


class OnlineSource(Protocol):
    """A subtitle service; see online.py."""

    report: dict[str, Any]  # what was searched and downloaded, for the JSON report

    def find(self) -> tuple[list[SubtitleCandidate], list[str]]:
        """Candidates, and notes on what could not be searched."""
        ...

    def fetch(self, candidate: SubtitleCandidate) -> str: ...


@dataclass(frozen=True)
class SourceResult:
    source: str
    candidates: tuple[SubtitleCandidate, ...]  # ranked, at most `max_candidates`
    notes: tuple[str, ...]  # what could not be searched, and why


def embedded_candidates(info: MediaInfo) -> list[SubtitleCandidate]:
    """Text subtitle streams in the file itself. Commentary tracks are skipped."""
    found = []
    for stream in info.subtitle_streams:
        if not stream.is_text_subtitle:
            continue
        title = stream.title or ""
        if stream.disposition.get("comment") or _COMMENTARY_TITLE.search(title):
            continue
        label = f"embedded #{stream.index}" + (f" {title!r}" if title else "")
        if stream.language:
            label += f" ({stream.language})"
        found.append(
            SubtitleCandidate(
                source="embedded",
                label=label,
                language=language_code(stream.language) if stream.language else None,
                hearing_impaired=bool(stream.disposition.get("hearing_impaired") or _SDH_TITLE.search(title)),
                forced=bool(stream.disposition.get("forced") or _FORCED_TITLE.search(title)),
                stream=stream.index,
                codec=stream.codec,
                bonus=5.0 if stream.is_default else 0.0,
            )
        )
    return found


def name_flags(tokens: Sequence[str]) -> tuple[str | None, bool, bool]:
    """(language, hearing impaired, forced) from file-name tokens such as ["en", "sdh"].

    "hi" means hearing impaired when another token names the language ("Movie.en.hi.srt"), and
    Hindi otherwise ("Movie.hi.srt")."""
    lowered = [token.lower() for token in tokens]
    codes = [(token, language_code(token)) for token in lowered]
    languages = [(token, code) for token, code in codes if code]
    hearing_impaired = any(token in _SDH_TOKENS for token in lowered)
    if len(languages) > 1 and any(token == "hi" for token, _ in languages):
        hearing_impaired = True
        languages = [(token, code) for token, code in languages if token != "hi"]
    forced = any(token in _FORCED_TOKENS for token in lowered)
    return (languages[0][1] if languages else None), hearing_impaired, forced


def _is_subtitle(path: Path) -> bool:
    return path.suffix.lower() in SIDECAR_SUFFIXES and path.is_file()


def _named_for(path: Path, stem: str) -> str | None:
    """The part of the name after the video's stem ("Movie.en.sdh.srt" -> ".en.sdh"), or None."""
    name = path.stem
    if not name.lower().startswith(stem.lower()):
        return None
    rest = name[len(stem) :]
    return rest if not rest or rest[0] in "._-" else None


def sidecar_candidates(video: Path) -> list[SubtitleCandidate]:
    """Subtitle files next to the video (DESIGN.md §6.3). They are searched in three layouts:

    - `Movie.en.srt` next to `Movie.mkv`, also inside a `Subs/` or `Subtitles/` folder
    - `Subs/Movie/2_English.srt` (TV season packs)
    - `Subs/English.srt`, when the video is the only one in its folder (movie releases)
    """
    folder = video.parent
    stem = video.stem
    found: list[tuple[Path, str]] = []  # (file, the part of its name that describes it)
    try:
        entries = sorted(folder.iterdir())
        for entry in entries:
            if _is_subtitle(entry) and (rest := _named_for(entry, stem)) is not None:
                found.append((entry, rest))
        only_video = sum(1 for e in entries if e.suffix.lower() in VIDEO_SUFFIXES and e.is_file()) <= 1
        for sub_dir in (e for e in entries if e.is_dir() and e.name.lower() in SUBTITLE_DIRS):
            for entry in sorted(sub_dir.iterdir()):
                if _is_subtitle(entry):
                    rest = _named_for(entry, stem)
                    if rest is not None:
                        found.append((entry, rest))
                    elif only_video:
                        found.append((entry, entry.stem))
                elif entry.is_dir() and entry.name.lower() == stem.lower():
                    found += [(file, file.stem) for file in sorted(entry.iterdir()) if _is_subtitle(file)]
    except OSError:
        pass  # an unreadable folder just has no sidecars

    candidates = []
    for path, description in found:
        language, hearing_impaired, forced = name_flags(_NAME_TOKEN.findall(description))
        candidates.append(
            SubtitleCandidate(
                source="sidecar",
                label=str(path.relative_to(folder)),
                language=language,
                hearing_impaired=hearing_impaired,
                forced=forced,
                path=path,
                bonus=5.0 if path.parent == folder else 0.0,
            )
        )
    return candidates


def explicit_candidate(path: Path) -> SubtitleCandidate:
    language, hearing_impaired, forced = name_flags(_NAME_TOKEN.findall(path.stem))
    return SubtitleCandidate(
        source="explicit",
        label=path.name,
        language=language,
        hearing_impaired=hearing_impaired,
        forced=forced,
        path=path,
    )


def score(
    candidate: SubtitleCandidate, languages: Sequence[str], prefer_hearing_impaired: bool
) -> float | None:
    """Higher is better; None filters the candidate out (wrong language, or a forced-only track)."""
    if candidate.forced:
        return None
    if candidate.language is None:
        value = 50.0  # unknown language: usable, but after the ones known to match
    else:
        positions = [i for i, code in enumerate(languages) if lang_matches(candidate.language, code)]
        if not positions:
            return None
        value = 100.0 - 10 * positions[0]
    if candidate.hearing_impaired == prefer_hearing_impaired:
        value += 20  # SDH tracks tend to be closer to verbatim
    return value + candidate.bonus


def rank(
    candidates: Sequence[SubtitleCandidate], languages: Sequence[str], prefer_hearing_impaired: bool
) -> list[SubtitleCandidate]:
    scored = []
    for candidate in candidates:
        value = score(candidate, languages, prefer_hearing_impaired)
        if value is not None:
            scored.append(replace(candidate, score=value))
    return sorted(scored, key=lambda c: -c.score)  # stable: ties keep discovery order


class SubtitleSearch:
    """The subtitle sources for one video, each searched only when it is reached: local sources come
    first, and the network is used only if none of them is usable (DESIGN.md §8.2)."""

    def __init__(
        self,
        info: MediaInfo,
        config: SubtitlesConfig,
        *,
        explicit: Path | None = None,
        online: OnlineSource | None = None,
    ) -> None:
        self.info = info
        self.config = config
        self.explicit = explicit
        self.online = online
        self._results: dict[str, SourceResult] = {}

    @property
    def searched(self) -> tuple[str, ...]:
        """The sources searched so far."""
        return tuple(self._results)

    @property
    def sources(self) -> tuple[str, ...]:
        """An explicit file alone, else the configured sources in order."""
        return ("explicit",) if self.explicit is not None else tuple(self.config.sources)

    def search(self, source: str) -> SourceResult:
        """Ranked candidates from one source; at most `max_candidates` are tried per source."""
        if source not in self._results:
            notes: list[str] = []
            if source == "explicit":
                assert self.explicit is not None
                found = [explicit_candidate(self.explicit)]
            elif source == "embedded":
                found = embedded_candidates(self.info)
            elif source == "sidecar":
                found = sidecar_candidates(self.info.path)
            elif self.online is None:
                found, notes = [], [f"{source}: not available"]
            else:
                found, notes = self.online.find()
            ranked = (
                found
                if source == "explicit"
                else rank(found, self.config.languages, self.config.prefer_hearing_impaired)
            )
            self._results[source] = SourceResult(
                source, tuple(ranked[: self.config.max_candidates]), tuple(notes)
            )
        return self._results[source]


def extracted_path(workdir: Path, stream: int, codec: str | None) -> Path:
    """Where stream `stream` is extracted to: ASS stays ASS, every other text codec becomes SRT."""
    return workdir / f"subtitles-{stream}.{'ass' if codec in ('ass', 'ssa') else 'srt'}"


def extract_subtitle_streams(
    ff: FFmpeg,
    media: Path,
    streams: Sequence[tuple[int, str | None]],
    workdir: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> dict[int, Path]:
    """Extract text subtitle streams, given as (index, codec), in one pass over the file: extracting
    means reading all of it. Streams already extracted into `workdir` are not extracted again."""
    outputs = {index: extracted_path(workdir, index, codec) for index, codec in streams}
    missing = {index: path for index, path in outputs.items() if not path.is_file()}
    if missing:
        args = ["-i", file_arg(media)]
        for index, path in missing.items():
            args += ["-map", f"0:{index}", "-f", path.suffix[1:], file_arg(path)]
        try:
            ff.run(args, on_progress=on_progress)
        except MediaError as exc:
            for path in missing.values():
                path.unlink(missing_ok=True)
            raise SubtitleError(f"could not extract the subtitle streams: {exc}") from exc
    return outputs


class SubtitleLoader:
    """Loads candidate text. Embedded streams are extracted in one pass over the file, the first time
    one of them is needed: extracting means reading the whole file."""

    def __init__(
        self,
        ff: FFmpeg,
        media: Path,
        workdir: Path,
        embedded: Sequence[SubtitleCandidate] = (),
        on_progress: Callable[[float], None] | None = None,
        online: OnlineSource | None = None,
        also: Sequence[tuple[int, str | None]] = (),
    ) -> None:
        self.ff = ff
        self.media = media
        self.workdir = workdir
        self.embedded = [c for c in embedded if c.stream is not None]
        self.on_progress = on_progress
        self.online = online
        self.also = list(also)  # (index, codec) of other text streams to extract in the same pass
        self._extracted: dict[int, Path] | None = None

    @property
    def pending(self) -> bool:
        """True while embedded streams still have to be extracted (a pass over the whole file)."""
        return self._extracted is None and bool(self.embedded)

    def _extract(self) -> dict[int, Path]:
        wanted = {c.stream: c.codec for c in self.embedded if c.stream is not None}
        for index, codec in self.also:
            wanted.setdefault(index, codec)
        return extract_subtitle_streams(
            self.ff, self.media, list(wanted.items()), self.workdir, on_progress=self.on_progress
        )

    def text(self, candidate: SubtitleCandidate) -> str:
        if candidate.source == "opensubtitles":
            if self.online is None:
                raise SubtitleError(f"{candidate.label}: OpenSubtitles is not available")
            return self.online.fetch(candidate)
        if candidate.path is not None:
            return read_subtitle_file(candidate.path, candidate.language)
        if candidate.stream is None:
            raise SubtitleError(f"{candidate.label}: nothing to load")
        if self._extracted is None:
            self._extracted = {}  # a failed extraction is not retried for the next stream
            self._extracted = self._extract()
        path = self._extracted.get(candidate.stream)
        if path is None or not path.exists():
            raise SubtitleError(f"{candidate.label}: stream was not extracted")
        return read_subtitle_file(path, candidate.language)
