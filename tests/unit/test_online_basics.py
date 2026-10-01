import random
from pathlib import Path
from typing import Any

import pytest

from video_beep_remover.config.loader import cache_root, load_config
from video_beep_remover.media.probe import parse_probe
from video_beep_remover.subtitles.cache import CachedSubtitle, SubtitleCache
from video_beep_remover.subtitles.names import (
    EpisodeNfo,
    episode_nfo,
    guess,
    guess_video,
    imdb_id_from_nfo,
    release_similarity,
)
from video_beep_remover.subtitles.online import OnlineSubtitles
from video_beep_remover.subtitles.opensubtitles import OnlineSubtitle
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


EPISODE_NFO = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<episodedetails>
    <title>Biscuit Bazooka</title>
    <showtitle>MythBusters</showtitle>
    <season>1</season>
    <episode>2</episode>
    <uniqueid type="imdb">tt0768454</uniqueid>
    <uniqueid type="tmdb" default="true">65242</uniqueid>
</episodedetails>
"""


def test_an_episode_named_without_its_show_finds_it_in_its_nfo(tmp_path: Path) -> None:
    video = tmp_path / "input" / "S01E02 - Biscuit Bazooka.mp4"
    video.parent.mkdir()
    assert guess(video.name).title == "Biscuit Bazooka"  # guessit takes the episode's title for the show's
    assert guess_video(video).title is None  # "input" names no show, so no title is searched
    video.with_suffix(".nfo").write_text(EPISODE_NFO, "utf-8")
    found = guess_video(video)
    assert (found.kind, found.title, found.season, found.episode) == ("episode", "MythBusters", 1, 2)
    assert episode_nfo(video) == EpisodeNfo("MythBusters", 768454)


def test_an_episode_in_a_library_takes_its_show_from_the_folders(tmp_path: Path) -> None:
    show = tmp_path / "MythBusters (2003)"
    season = show / "Season 01"
    season.mkdir(parents=True)
    assert guess_video(season / "S01E02 - Biscuit Bazooka.mp4").title == "MythBusters"
    (show / "tvshow.nfo").write_text("<tvshow><title>Mythbusters</title></tvshow>", "utf-8")
    assert guess_video(season / "S01E02 - Biscuit Bazooka.mp4").title == "Mythbusters"  # the .nfo's
    assert guess_video(season / "Show Name - S01E02 - Pilot.mkv").title == "Show Name"  # the name has one
    assert guess_video(tmp_path / "The.Movie.2019.mkv").title == "The Movie"


class Client:
    """OpenSubtitles' search, with results for some kinds of search ("imdb_id", "query")."""

    def __init__(self, results: dict[str, list[OnlineSubtitle]]) -> None:
        self.results = results
        self.searches: list[dict[str, Any]] = []

    def search(self, *, languages: list[str], **params: Any) -> list[OnlineSubtitle]:
        self.searches.append(params)
        return self.results.get(next(iter(params)), [])


def test_an_episode_is_searched_by_its_imdb_id_then_by_its_show(tmp_path: Path) -> None:
    video = tmp_path / "S01E02 - Biscuit Bazooka.mp4"
    video.write_bytes(b"")  # too small for a movie hash
    video.with_suffix(".nfo").write_text(EPISODE_NFO, "utf-8")
    config = load_config(None, env={}, cwd=tmp_path).config
    info = parse_probe(video, {"format": {"duration": "3003"}, "streams": []})

    def online(client: Client) -> OnlineSubtitles:
        return OnlineSubtitles(config, info, client=client, cache=SubtitleCache(tmp_path))  # type: ignore[arg-type]

    by_id = Client({"imdb_id": [OnlineSubtitle(1, "1.srt", "en", False, False, False, None, None, 9, False)]})
    assert online(by_id).planned_searches() == [
        {"imdb_id": 768454},
        {"query": "MythBusters", "season": 1, "episode": 2},
    ]
    candidates, _ = online(by_id).find()
    assert [c.file_id for c in candidates] == [1]
    assert by_id.searches == [{"imdb_id": 768454}]  # found by the id: the title is not searched
    nothing = Client({})
    online(nothing).find()
    assert [next(iter(s)) for s in nothing.searches] == ["imdb_id", "query"]
