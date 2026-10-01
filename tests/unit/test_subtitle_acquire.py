from pathlib import Path
from typing import Any

import pytest

from video_beep_remover.config.schema import SubtitlesConfig
from video_beep_remover.media.probe import parse_probe
from video_beep_remover.subtitles.acquire import (
    SubtitleCandidate,
    SubtitleSearch,
    embedded_candidates,
    name_flags,
    rank,
    sidecar_candidates,
)


def touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def media(path: Path, *streams: dict[str, Any]) -> Any:
    raw = [{"index": 0, "codec_type": "video", "codec_name": "h264", "avg_frame_rate": "24000/1001"}]
    raw += [{"index": i + 1, "codec_type": "subtitle", **s} for i, s in enumerate(streams)]
    return parse_probe(path, {"format": {"duration": "60"}, "streams": raw})


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["en"], ("en", False, False)),
        (["eng", "sdh"], ("en", True, False)),
        (["English", "forced"], ("en", False, True)),
        (["2", "english"], ("en", False, False)),
        (["en", "hi"], ("en", True, False)),  # "hi" next to a language means hearing impaired
        (["hi"], ("hi", False, False)),  # ... and Hindi on its own
        (["cc"], (None, True, False)),
        ([], (None, False, False)),
    ],
)
def test_name_flags(tokens: list[str], expected: tuple[str | None, bool, bool]) -> None:
    assert name_flags(tokens) == expected


def test_sidecars_next_to_the_video_and_in_subs_folders(tmp_path: Path) -> None:
    video = tmp_path / "Movie (2019).mkv"
    touch(
        video,
        tmp_path / "Movie (2019).srt",
        tmp_path / "Movie (2019).en.sdh.srt",
        tmp_path / "Movie (2019).fr.ASS",
        tmp_path / "Movie (2019) Extras.srt",  # another video's subtitles
        tmp_path / "Other.en.srt",
        tmp_path / "Movie (2019).en.txt",
        tmp_path / "Subs" / "Movie (2019).de.vtt",
        tmp_path / "Subs" / "2_English.srt",  # the only video in the folder: any file in Subs/ is for it
    )
    found = {c.label: (c.language, c.hearing_impaired) for c in sidecar_candidates(video)}
    assert found == {
        "Movie (2019).en.sdh.srt": ("en", True),
        "Movie (2019).fr.ASS": ("fr", False),
        "Movie (2019).srt": (None, False),
        str(Path("Subs") / "2_English.srt"): ("en", False),
        str(Path("Subs") / "Movie (2019).de.vtt"): ("de", False),
    }


def test_review_subtitles_are_not_sidecars(tmp_path: Path) -> None:
    """vbr's own review subtitles (a scan's, and a clean's next to its output) name what was muted."""
    video = tmp_path / "Movie.mkv"
    touch(
        video, tmp_path / "Movie.en.srt", tmp_path / "Movie.review.srt", tmp_path / "Movie.clean.review.srt"
    )
    assert [c.label for c in sidecar_candidates(video)] == ["Movie.en.srt"]


def test_season_pack_layout(tmp_path: Path) -> None:
    episode = tmp_path / "Show.S01E02.mkv"
    touch(
        episode,
        tmp_path / "Show.S01E01.mkv",
        tmp_path / "Subs" / "Show.S01E02" / "3_English.srt",
        tmp_path / "Subs" / "Show.S01E01" / "3_English.srt",
        tmp_path / "Subs" / "readme.srt",  # several videos here: not claimed by any of them
    )
    [candidate] = sidecar_candidates(episode)
    assert candidate.label == str(Path("Subs") / "Show.S01E02" / "3_English.srt")
    assert candidate.language == "en"


def test_embedded_text_streams_only(tmp_path: Path) -> None:
    info = media(
        tmp_path / "movie.mkv",
        {"codec_name": "subrip", "tags": {"language": "eng"}, "disposition": {"default": 1}},
        {"codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
        {"codec_name": "ass", "tags": {"language": "eng", "title": "English SDH"}},
        {"codec_name": "subrip", "tags": {"language": "eng", "title": "Forced"}},
        {"codec_name": "mov_text", "tags": {"language": "eng", "title": "Director's Commentary"}},
        {"codec_name": "subrip", "disposition": {"hearing_impaired": 1}},
    )
    found = [(c.stream, c.language, c.hearing_impaired, c.forced) for c in embedded_candidates(info)]
    assert found == [
        (1, "en", False, False),
        (3, "en", True, False),
        (4, "en", False, True),
        (6, None, True, False),
    ]


def candidate(label: str, language: str | None, **kwargs: Any) -> SubtitleCandidate:
    return SubtitleCandidate(source="sidecar", label=label, language=language, **kwargs)


def test_ranking_filters_and_orders() -> None:
    ranked = rank(
        [
            candidate("plain", "en"),
            candidate("unknown", None),
            candidate("sdh", "en", hearing_impaired=True),
            candidate("french", "fr"),
            candidate("forced", "en", forced=True),
            candidate("german", "de"),
        ],
        languages=["en", "de"],
        prefer_hearing_impaired=True,
    )
    assert [c.label for c in ranked] == ["sdh", "plain", "german", "unknown"]
    plain_first = rank(
        [candidate("sdh", "en", hearing_impaired=True), candidate("plain", "en")], ["en"], False
    )
    assert [c.label for c in plain_first] == ["plain", "sdh"]


class FakeOnline:
    def __init__(self) -> None:
        self.searched = 0
        self.report: dict[str, Any] = {}

    def find(self) -> tuple[list[SubtitleCandidate], list[str]]:
        self.searched += 1
        found = [SubtitleCandidate("opensubtitles", f"OpenSubtitles #{n}", "en", file_id=n) for n in range(5)]
        return found, ["OpenSubtitles: a note"]

    def fetch(self, candidate: SubtitleCandidate) -> str:
        raise AssertionError("not used here")


def test_sources_are_searched_in_order_and_only_when_reached(tmp_path: Path) -> None:
    video = tmp_path / "movie.mkv"
    touch(video, tmp_path / "movie.en.srt")
    info = media(video, {"codec_name": "subrip", "tags": {"language": "eng"}})
    online = FakeOnline()
    search = SubtitleSearch(info, SubtitlesConfig(), online=online)
    assert search.sources == ("embedded", "sidecar", "opensubtitles")
    assert [c.source for c in search.search("embedded").candidates] == ["embedded"]
    assert search.searched == ("embedded",) and online.searched == 0  # the network is not touched yet
    result = search.search("opensubtitles")
    assert [c.file_id for c in result.candidates] == [0, 1, 2]  # max_candidates per source
    assert result.notes == ("OpenSubtitles: a note",)
    search.search("opensubtitles")
    assert online.searched == 1  # searched once per video

    capped = SubtitleSearch(info, SubtitlesConfig(sources=["sidecar", "embedded"], max_candidates=1))
    assert capped.sources == ("sidecar", "embedded")
    assert capped.search("opensubtitles").notes == ("opensubtitles: not available",)


def test_explicit_file_skips_the_search(tmp_path: Path) -> None:
    video = tmp_path / "movie.mkv"
    touch(video, tmp_path / "movie.en.srt")
    info = media(video, {"codec_name": "subrip", "tags": {"language": "eng"}})
    search = SubtitleSearch(info, SubtitlesConfig(), explicit=tmp_path / "mine.fr.srt", online=FakeOnline())
    assert search.sources == ("explicit",)
    [only] = search.search("explicit").candidates
    assert (only.source, only.label, only.language, only.trusted) == ("explicit", "mine.fr.srt", "fr", True)
