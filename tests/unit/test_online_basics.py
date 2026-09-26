import random
from pathlib import Path

import pytest

from video_beep_remover.config.loader import cache_root, load_config
from video_beep_remover.subtitles.cache import CachedSubtitle, SubtitleCache
from video_beep_remover.subtitles.names import guess, imdb_id_from_nfo, release_similarity
from video_beep_remover.subtitles.oshash import fingerprint, opensubtitles_hash


def reference_hash(data: bytes) -> str:
    """The OpenSubtitles hash written out plainly, to check the implementation against."""
    total = len(data)
    for block in (data[:65536], data[-65536:]):
        for i in range(0, 65536, 8):
            total += int.from_bytes(block[i : i + 8], "little")
    return f"{total % 2**64:016x}"


@pytest.mark.parametrize(
    "data",
    [
        bytes(131072),  # exactly 128 KiB of zeros: the hash is the size
        b"\xff" * 200_000,  # every word is 2**64 - 1: the sum wraps around
        random.Random(1).randbytes(1_000_003),  # the two ends do not overlap
    ],
    # Short ids: pytest puts the test id in an environment variable, which Windows limits to 32767 characters.
    ids=["zeros", "wrap-around", "random"],
)
def test_hash_matches_the_reference_algorithm(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "movie.mkv"
    path.write_bytes(data)
    assert opensubtitles_hash(path) == reference_hash(data)
    assert fingerprint(path) == f"{reference_hash(data)}-{len(data)}"


def test_small_files_have_no_hash(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mkv"
    path.write_bytes(bytes(131071))
    assert opensubtitles_hash(path) is None and fingerprint(path) is None


def test_file_names_give_title_year_episode_and_release() -> None:
    movie = guess("The.Movie.2019.1080p.BluRay.x264-SPARKS.mkv")
    assert (movie.kind, movie.title, movie.year) == ("movie", "The Movie", 2019)
    assert movie.release["release_group"] == "sparks"
    episode = guess("Show Name - S01E02 - Pilot.mkv")
    assert (episode.kind, episode.title, episode.season, episode.episode) == ("episode", "Show Name", 1, 2)
    assert release_similarity(movie, guess("The.Movie.2019.720p.BluRay.x264-SPARKS")) == 16
    assert release_similarity(movie, guess("The.Movie.2019.WEBRip.x265-OTHER")) == 0


def test_imdb_id_from_a_kodi_nfo(tmp_path: Path) -> None:
    video = tmp_path / "The Movie (2019).mkv"
    assert imdb_id_from_nfo(video) is None
    (tmp_path / "movie.nfo").write_text("<movie><imdbid>tt0012345</imdbid></movie>", "utf-8")
    assert imdb_id_from_nfo(video) == 12345
    video.with_suffix(".nfo").write_text("https://www.imdb.com/title/tt7654321/", "utf-8")
    assert imdb_id_from_nfo(video) == 7654321  # the file's own .nfo wins


def test_cache_keeps_files_and_an_index_per_video(tmp_path: Path) -> None:
    cache = SubtitleCache(tmp_path)
    assert cache.path(42) is None and cache.known("abc-1") == [] and cache.usage() == (0, 0)
    stored = cache.store(42, b"1\n00:00:01,000 --> 00:00:02,000\nHi\n", ".SRT")
    assert stored.name == "42.srt" and cache.path(42) == stored
    entry = CachedSubtitle(42, "42.srt", "en", True, 23.976, "The.Movie.2019", True, 900)
    cache.remember("abc-1", entry)
    cache.remember("abc-1", entry)  # remembered once
    assert cache.known("abc-1") == [entry]
    assert cache.known("other-2") == []
    assert cache.usage() == (1, stored.stat().st_size)
    stored.unlink()
    assert cache.known("abc-1") == []  # the file is gone, so the entry is useless
    cache.store(7, b"x")
    assert cache.clear() == 2  # the file and the index
    assert cache.usage() == (0, 0)


def test_unreadable_index_is_ignored(tmp_path: Path) -> None:
    cache = SubtitleCache(tmp_path)
    cache.dir.mkdir(parents=True)
    (cache.dir / "index.json").write_text("{not json", "utf-8")
    assert cache.known("abc-1") == []
    cache.remember("abc-1", CachedSubtitle(1, "1.srt"))  # rewritten from scratch


def test_cache_root_follows_the_config(tmp_path: Path) -> None:
    config = load_config(None, env={}, cwd=tmp_path).config
    assert cache_root(config).name == "cache"  # the tests' stand-in for the per-user cache dir
    custom = load_config(None, env={}, cwd=tmp_path, overrides={"cache.dir": str(tmp_path / "c")}).config
    assert cache_root(custom) == tmp_path / "c"
