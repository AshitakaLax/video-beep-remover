"""Downloaded subtitles, kept so a subtitle never costs download quota twice (DESIGN.md §8.3).

    <cache>/subtitles/<provider>/<file id>.<ext>
    <cache>/subtitles/<provider>/index.json      fingerprint -> what was downloaded for that video

The index lets a later run, even an offline one or one without an API key, reuse what an earlier
run downloaded for the same video."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

INDEX = "index.json"
INDEX_VERSION = 1


@dataclass(frozen=True)
class CachedSubtitle:
    """What is known about a downloaded subtitle, from the search that found it."""

    file_id: int
    file: str  # name inside the provider's cache directory
    language: str | None = None
    hearing_impaired: bool = False
    fps: float | None = None
    release: str | None = None
    moviehash_match: bool = False
    download_count: int = 0


class SubtitleCache:
    def __init__(self, root: Path, provider: str = "opensubtitles") -> None:
        self.dir = root / "subtitles" / provider

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _index(self) -> dict[str, Any]:
        try:
            data = json.loads((self.dir / INDEX).read_text("utf-8"))
            if isinstance(data, dict) and data.get("version") == INDEX_VERSION:
                return data
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.warning("ignoring the unreadable subtitle cache index %s: %s", self.dir / INDEX, exc)
        return {"version": INDEX_VERSION, "media": {}}

    def path(self, file_id: int) -> Path | None:
        """The cached file for `file_id`, if there is one."""
        if not self.dir.is_dir():
            return None
        return next((p for p in sorted(self.dir.glob(f"{file_id}.*")) if not p.name.endswith(".tmp")), None)

    def store(self, file_id: int, data: bytes, suffix: str = ".srt") -> Path:
        path = self.dir / f"{file_id}{suffix.lower() if suffix.startswith('.') else '.srt'}"
        self._write(path, data)
        return path

    def remember(self, fingerprint: str, entry: CachedSubtitle) -> None:
        """Record that `entry` was downloaded for the video with this fingerprint."""
        index = self._index()
        entries = [e for e in index["media"].get(fingerprint, []) if e.get("file_id") != entry.file_id]
        index["media"][fingerprint] = [*entries, asdict(entry)]
        self._write(self.dir / INDEX, (json.dumps(index, indent=1, sort_keys=True) + "\n").encode("utf-8"))

    def known(self, fingerprint: str) -> list[CachedSubtitle]:
        """Subtitles downloaded earlier for this video whose files are still cached."""
        known = []
        for raw in self._index()["media"].get(fingerprint, []):
            try:
                entry = CachedSubtitle(**raw)
            except TypeError:
                continue
            if (self.dir / entry.file).is_file():
                known.append(entry)
        return known

    def usage(self) -> tuple[int, int]:
        """(number of cached subtitle files, their total size in bytes)."""
        if not self.dir.is_dir():
            return 0, 0
        files = [p for p in self.dir.iterdir() if p.is_file() and p.name != INDEX]
        return len(files), sum(p.stat().st_size for p in files)

    def clear(self) -> int:
        """Delete every cached subtitle and the index; returns the number of files removed."""
        if not self.dir.is_dir():
            return 0
        removed = 0
        for path in self.dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
        return removed
