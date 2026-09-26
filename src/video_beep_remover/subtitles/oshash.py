"""The OpenSubtitles movie hash (DESIGN.md §6.4) and the file fingerprint built on it (§8.3)."""

import struct
from pathlib import Path

CHUNK = 64 * 1024
_MASK = 0xFFFF_FFFF_FFFF_FFFF


def opensubtitles_hash(path: Path) -> str | None:
    """File size + the first and last 64 KiB summed as little-endian uint64, mod 2**64.

    It reads only 128 KiB, so it is cheap even for 50 GB files. Files smaller than 128 KiB have no
    hash (None): OpenSubtitles then has to be searched by title."""
    size = path.stat().st_size
    if size < 2 * CHUNK:
        return None
    value = size
    with path.open("rb") as file:
        for offset in (0, size - CHUNK):
            file.seek(offset)
            for (word,) in struct.iter_unpack("<Q", file.read(CHUNK)):
                value = (value + word) & _MASK
    return f"{value:016x}"


def fingerprint(path: Path) -> str | None:
    """A cheap identity for a media file: its OpenSubtitles hash plus its size."""
    movie_hash = opensubtitles_hash(path)
    return None if movie_hash is None else f"{movie_hash}-{path.stat().st_size}"
