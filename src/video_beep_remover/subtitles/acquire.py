"""Find subtitle candidates for a video, rank them and load their text (DESIGN.md §6.3)."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from video_beep_remover.config.schema import SubtitlesConfig
from video_beep_remover.errors import MediaError, SubtitleError
from video_beep_remover.languages import lang_matches, language_code
from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import VIDEO_SUFFIXES, MediaInfo
from video_beep_remover.subtitles.parse import read_subtitle_file

CandidateSource = Literal["explicit", "embedded", "sidecar"]

# Text subtitle codecs FFmpeg can convert to SRT or ASS. Image-based ones (PGS, VobSub, DVB) would need OCR.
TEXT_CODECS = frozenset({"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"})
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


@dataclass(frozen=True)
class SubtitleSearch:
    candidates: tuple[SubtitleCandidate, ...]  # in the order they are tried
    notes: tuple[str, ...]  # sources that were skipped, and why


def embedded_candidates(info: MediaInfo) -> list[SubtitleCandidate]:
    """Text subtitle streams in the file itself. Commentary tracks are skipped."""
    found = []
    for stream in info.subtitle_streams:
        if (stream.codec or "") not in TEXT_CODECS:
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


def find_candidates(
    info: MediaInfo, config: SubtitlesConfig, *, offline: bool, explicit: Path | None = None
) -> SubtitleSearch:
    """Candidates in the order to try them: an explicit file alone, else each configured source in
    turn, ranked within the source. At most `max_candidates` are returned."""
    if explicit is not None:
        return SubtitleSearch((explicit_candidate(explicit),), ())
    candidates: list[SubtitleCandidate] = []
    notes: list[str] = []
    for source in config.sources:
        if source == "embedded":
            found = embedded_candidates(info)
        elif source == "sidecar":
            found = sidecar_candidates(info.path)
        else:  # opensubtitles
            notes.append(
                "OpenSubtitles: skipped (offline)"
                if offline
                else "OpenSubtitles: online search is not implemented yet (milestone M3)"
            )
            continue
        candidates += rank(found, config.languages, config.prefer_hearing_impaired)
    return SubtitleSearch(tuple(candidates[: config.max_candidates]), tuple(notes))


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
    ) -> None:
        self.ff = ff
        self.media = media
        self.workdir = workdir
        self.embedded = [c for c in embedded if c.stream is not None]
        self.on_progress = on_progress
        self._extracted: dict[int, Path] | None = None

    @property
    def pending(self) -> bool:
        """True while embedded streams still have to be extracted (a pass over the whole file)."""
        return self._extracted is None and bool(self.embedded)

    def _extract(self) -> dict[int, Path]:
        outputs: dict[int, Path] = {}
        args = ["-i", file_arg(self.media)]
        for candidate in self.embedded:
            assert candidate.stream is not None
            fmt = "ass" if candidate.codec in ("ass", "ssa") else "srt"
            target = self.workdir / f"subtitles-{candidate.stream}.{fmt}"
            args += ["-map", f"0:{candidate.stream}", "-f", fmt, file_arg(target)]
            outputs[candidate.stream] = target
        if outputs:
            try:
                self.ff.run(args, on_progress=self.on_progress)
            except MediaError as exc:
                raise SubtitleError(f"could not extract the subtitle streams: {exc}") from exc
        return outputs

    def text(self, candidate: SubtitleCandidate) -> str:
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
