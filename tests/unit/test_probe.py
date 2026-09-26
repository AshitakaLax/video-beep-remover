from pathlib import Path
from typing import Any

import pytest

from video_beep_remover.errors import MediaError, UsageError
from video_beep_remover.languages import lang_matches
from video_beep_remover.media.ffmpeg import FFmpegVersion
from video_beep_remover.media.probe import parse_probe, select_audio_stream


def audio(index: int, language: str | None = "eng", channels: int = 2, **extra: Any) -> dict[str, Any]:
    tags = {"language": language} if language else {}
    if "title" in extra:
        tags["title"] = extra.pop("title")
    return {"index": index, "codec_type": "audio", "codec_name": "aac", "channels": channels, "tags": tags,
            "disposition": extra.pop("disposition", {"default": 0}), **extra}  # fmt: skip


def media(*streams: dict[str, Any], duration: str | None = "60.5") -> Any:
    fmt = {"format_name": "matroska,webm", "size": "1234"}
    if duration:
        fmt["duration"] = duration
    video = {"index": 0, "codec_type": "video", "codec_name": "h264", "disposition": {"default": 1}}
    return parse_probe(Path("movie.mkv"), {"format": fmt, "streams": [video, *streams]})


def test_parse_probe_reads_streams_tags_and_dispositions() -> None:
    info = media(audio(1, title="Commentary", disposition={"comment": 1}), audio(2, language="und"))
    assert info.duration == 60.5 and info.size == 1234
    first, second = info.audio_streams
    assert first.is_commentary and first.title == "Commentary"
    assert second.language is None  # "und" means untagged


def test_duration_falls_back_to_streams_and_must_exist() -> None:
    info = media(audio(1, duration="12.0"), duration=None)
    assert info.duration == 12.0
    with pytest.raises(MediaError, match="duration"):
        media(audio(1), duration=None)


def test_selection_skips_commentary_and_description_and_prefers_language() -> None:
    info = media(
        audio(1, title="Director's commentary", channels=6),
        audio(2, disposition={"visual_impaired": 1}),
        audio(3, language="fra", channels=6, disposition={"default": 1}),
        audio(4, channels=2),
    )
    assert select_audio_stream(info, "en").index == 4
    assert select_audio_stream(info, "fr").index == 3


def test_selection_prefers_default_then_channels_then_index() -> None:
    info = media(audio(1, channels=2), audio(2, channels=6), audio(3, channels=6, disposition={"default": 1}))
    assert select_audio_stream(info, "en").index == 3
    info = media(audio(1, channels=2), audio(2, channels=6))
    assert select_audio_stream(info, "en").index == 2


def test_explicit_stream_index() -> None:
    info = media(audio(1), audio(2))
    assert select_audio_stream(info, "en", 2).index == 2
    with pytest.raises(UsageError, match="video"):
        select_audio_stream(info, "en", 0)
    with pytest.raises(UsageError, match="no stream #9"):
        select_audio_stream(info, "en", 9)


def test_no_audio_is_an_error() -> None:
    with pytest.raises(MediaError, match="no audio"):
        select_audio_stream(media(), "en")


@pytest.mark.parametrize(
    ("tag", "code", "expected"),
    [("eng", "en", True), ("en", "en", True), ("fre", "fr", True), ("fra", "fr", True), ("por", "pt-BR", True),
     ("deu", "en", False), (None, "en", False)],
)  # fmt: skip
def test_lang_matches(tag: str | None, code: str, expected: bool) -> None:
    assert lang_matches(tag, code) is expected


@pytest.mark.parametrize(
    ("banner", "major", "minor", "number"),
    [
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023", 6, 1, "6.1.1-3ubuntu5"),
        ("ffmpeg version 5.1.1-static https://johnvansickle.com/ffmpeg/", 5, 1, "5.1.1-static"),
        ("ffmpeg version n7.0.2 Copyright", 7, 0, "n7.0.2"),
        ("ffmpeg version N-112345-gabcdef Copyright", None, None, "N-112345-gabcdef"),
    ],
)
def test_ffmpeg_version_parsing(banner: str, major: int | None, minor: int | None, number: str) -> None:
    version = FFmpegVersion.parse(banner)
    assert (version.major, version.minor, version.number) == (major, minor, number)
    assert version.at_least(5, 1)
    assert version.at_least(7) is (major is None or major >= 7)
