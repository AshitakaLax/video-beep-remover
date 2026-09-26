"""Transcripts and speech regions kept between runs (DESIGN.md §8.3).

    <cache>/asr/<fingerprint>/<stream>/<model>-<settings hash>.jsonl   one transcribed span per line
    <cache>/asr/<fingerprint>/<stream>/speech.json                     speech regions (hybrid)

The fingerprint identifies the file (OpenSubtitles hash and size), the settings hash everything that
changes what the model hears: model, precision, language, beam size, prompt setting and VAD. Re-running
a file, e.g. after editing the word list, then needs little or no speech recognition: a window that a
cached span covers, or that a cached whole-track transcript covers, is served from here. Least recently
used files are deleted when the transcripts outgrow cache.max_size_gb."""

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_beep_remover.models import Word

log = logging.getLogger(__name__)

VERSION = 2  # 2: anchor words that start in a pause are moved to where speech resumes
EDGE_S = 0.3  # words this close to a transcribed clip's edge are unreliable (as in detect/confirm.py)
TOLERANCE_S = 0.001
SPEECH = "speech.json"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Transcript:
    """The words heard in a clip. A clean edge was trimmed to silence, so no word is cut there."""

    start: float
    end: float
    words: tuple[Word, ...]
    clean_start: bool = False
    clean_end: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def settings_key(**settings: Any) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _words(raw: Any) -> tuple[Word, ...]:
    return tuple(Word(str(t), float(s), float(e), float(p)) for t, s, e, p in raw)


class TranscriptStore:
    """The transcribed spans of one audio stream, by one model with one set of settings."""

    def __init__(self, path: Path, duration: float) -> None:
        self.path = path
        self.duration = duration
        # requested window (ms) -> (the window, its transcript); the clip may be trimmed to speech
        self._windows: dict[tuple[int, int], tuple[tuple[float, float], Transcript]] = {}
        self._track: Transcript | None = None
        self._loaded = False

    @staticmethod
    def _ms(start: float, end: float) -> tuple[int, int]:
        return round(start * 1000), round(end * 1000)

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            lines = self.path.read_text("utf-8").splitlines()
            os.utime(self.path)  # recently used: evicted last
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("ignoring the unreadable transcript cache %s: %s", self.path, exc)
            return
        for line in lines:
            try:
                entry = json.loads(line)
                clip = Transcript(
                    float(entry["clip"][0]),
                    float(entry["clip"][1]),
                    _words(entry["words"]),
                    bool(entry.get("clean", (False, False))[0]),
                    bool(entry.get("clean", (False, False))[1]),
                )
            except (ValueError, KeyError, TypeError, IndexError):
                continue  # a line cut short by an interrupted run, say
            if entry.get("track"):
                self._track = clip
            elif isinstance(entry.get("window"), list):
                window = (float(entry["window"][0]), float(entry["window"][1]))
                self._windows[self._ms(*window)] = (window, clip)

    def _reliable(self, window: tuple[float, float], clip: Transcript) -> tuple[float, float]:
        """The part of `window` its transcript covers reliably. A clean (trimmed) edge vouches for the
        silence trimmed off; any other edge may cut a word, except at the ends of the file."""
        if clip.clean_start:
            start = min(window[0], clip.start)
        else:
            start = clip.start if clip.start <= TOLERANCE_S else clip.start + EDGE_S
        if clip.clean_end:
            end = max(window[1], clip.end)
        else:
            end = clip.end if clip.end >= self.duration - TOLERANCE_S else clip.end - EDGE_S
        return start, end

    def _entries(self) -> list[tuple[tuple[float, float], Transcript]]:
        track = [((self._track.start, self._track.end), self._track)] if self._track else []
        return track + list(self._windows.values())

    def track(self) -> list[Word] | None:
        """The whole track's words, if it was transcribed in one go (the full strategy)."""
        self._load()
        return list(self._track.words) if self._track else None

    def window(self, start: float, end: float) -> Transcript | None:
        """What was heard in the window [start, end]: the transcript of that very window, or the words
        a longer transcript heard there (its edges are then clean: no word is cut at them)."""
        self._load()
        exact = self._windows.get(self._ms(start, end))
        if exact is not None:
            return exact[1]
        for window, clip in self._entries():
            low, high = self._reliable(window, clip)
            if low - TOLERANCE_S <= start and end <= high + TOLERANCE_S:
                words = tuple(w for w in clip.words if start <= w.start < end)
                return Transcript(start, end, words, True, True)
        return None

    def cover(self, start: float, end: float) -> tuple[list[Transcript], list[tuple[float, float]]]:
        """Cached transcripts for the window [start, end], cut to it, and the stretches of the window
        none of them covers reliably, which still need transcribing. A cut edge is clean when it lies
        in the reliable part of its transcript: no word is cut there."""
        found = self.window(start, end)
        if found is not None:
            return [found], []
        pieces: list[Transcript] = []
        covered: list[tuple[float, float]] = []
        for window, clip in self._windows.values():
            low, high = self._reliable(window, clip)
            # At the window's own edges a fresh transcription would be no more reliable than the clip.
            reach_low = min(low, start) if clip.start <= start + TOLERANCE_S else low
            reach_high = max(high, end) if clip.end >= end - TOLERANCE_S else high
            if min(reach_high, end) - max(reach_low, start) <= TOLERANCE_S:
                continue
            covered.append((max(reach_low, start), min(reach_high, end)))
            a, b = max(clip.start, start), min(clip.end, end)
            clean_start = clip.clean_start if a <= clip.start else a >= low - TOLERANCE_S
            clean_end = clip.clean_end if b >= clip.end else b <= high + TOLERANCE_S
            words = tuple(w for w in clip.words if a <= w.start < b)
            pieces.append(Transcript(a, b, words, clean_start, clean_end))
        gaps: list[tuple[float, float]] = []
        position = start
        for low, high in sorted(covered):
            if low - position > TOLERANCE_S:
                gaps.append((position, low))
            position = max(position, high)
        if end - position > TOLERANCE_S:
            gaps.append((position, end))
        return sorted(pieces, key=lambda t: t.start), gaps

    def _append(self, entry: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as exc:
            log.warning("could not write the transcript cache %s: %s", self.path, exc)

    @staticmethod
    def _entry(clip: Transcript) -> dict[str, Any]:
        return {
            "clip": [round(clip.start, 4), round(clip.end, 4)],
            "clean": [clip.clean_start, clip.clean_end],
            "words": [
                [w.text, round(w.start, 3), round(w.end, 3), round(w.probability, 3)] for w in clip.words
            ],
        }

    def add_window(self, start: float, end: float, clip: Transcript) -> None:
        """Remember the transcript of the window [start, end] (its clip may be trimmed to speech)."""
        self._load()
        self._windows[self._ms(start, end)] = ((start, end), clip)
        self._append({"window": [round(start, 4), round(end, 4)], **self._entry(clip)})

    def add_track(self, words: Sequence[Word], end: float) -> None:
        """Remember the whole track's transcript. It covers the whole file, even if the decoded audio
        ended a little before the container's duration."""
        self._load()
        self._track = Transcript(0.0, max(end, self.duration), tuple(words), True, True)
        self._append({"track": True, **self._entry(self._track)})


class TranscriptCache:
    def __init__(self, root: Path) -> None:
        self.dir = root / "asr"

    def _folder(self, fingerprint: str, stream: int) -> Path:
        return self.dir / fingerprint / str(stream)

    def store(self, fingerprint: str, stream: int, model: str, key: str, duration: float) -> TranscriptStore:
        name = _UNSAFE.sub("_", Path(model).name or model)[:60]
        return TranscriptStore(self._folder(fingerprint, stream) / f"{name}-{key}.jsonl", duration)

    def speech(self, fingerprint: str, stream: int) -> list[tuple[float, float]] | None:
        path = self._folder(fingerprint, stream) / SPEECH
        try:
            data = json.loads(path.read_text("utf-8"))
            if data.get("version") != VERSION:
                return None
            regions = [(float(start), float(end)) for start, end in data["regions"]]
            os.utime(path)
            return regions
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("ignoring the unreadable speech cache %s: %s", path, exc)
            return None

    def remember_speech(self, fingerprint: str, stream: int, regions: Sequence[tuple[float, float]]) -> None:
        path = self._folder(fingerprint, stream) / SPEECH
        data = {"version": VERSION, "regions": [[round(s, 3), round(e, 3)] for s, e in regions]}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{SPEECH}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data), "utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not write the speech cache %s: %s", path, exc)

    def _files(self) -> list[Path]:
        return [p for p in self.dir.rglob("*") if p.is_file()] if self.dir.is_dir() else []

    def usage(self) -> tuple[int, int]:
        """(number of files, total size in bytes)."""
        files = self._files()
        return len(files), sum(p.stat().st_size for p in files)

    def evict(self, limit: int) -> int:
        """Delete least recently used files until the transcripts take at most `limit` bytes."""
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        removed = 0
        for path in files:
            if total <= limit:
                break
            size = path.stat().st_size
            with contextlib.suppress(OSError):
                path.unlink()
                total -= size
                removed += 1
        return removed

    def clear(self) -> int:
        files = len(self._files())
        shutil.rmtree(self.dir, ignore_errors=True)
        return files
