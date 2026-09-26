from pathlib import Path
from typing import Any

import pytest

from video_beep_remover.config.schema import OutputConfig
from video_beep_remover.errors import DependencyError
from video_beep_remover.media.ffmpeg import FFmpeg, FFmpegVersion
from video_beep_remover.media.probe import MediaInfo, StreamInfo
from video_beep_remover.media.render import (
    build_command,
    choose_encoder,
    command_file,
    disposition_value,
    normalize_intervals,
    plan_streams,
)
from video_beep_remover.models import CensorInterval as Span

ENCODERS = frozenset({"aac", "ac3", "eac3", "flac", "libopus", "libmp3lame", "pcm_s16le"})


def stub_ffmpeg(version: str = "ffmpeg version 6.1.1", encoders: frozenset[str] = ENCODERS) -> FFmpeg:
    ff = FFmpeg.__new__(FFmpeg)
    ff.__dict__.update(version=FFmpegVersion.parse(version), encoders=encoders)
    return ff


def stream(index: int, kind: str = "audio", **fields: Any) -> StreamInfo:
    return StreamInfo(index=index, kind=kind, **fields)  # type: ignore[arg-type]


def info(*streams: StreamInfo, name: str = "movie.mkv") -> MediaInfo:
    return MediaInfo(Path(name), "matroska", 100.0, 0.0, 1, streams)


def test_normalize_intervals_merges_close_and_lengthens_short_spans() -> None:
    spans = [Span(5.0, 5.01), Span(1.0, 2.0), Span(2.04, 3.0), Span(-1.0, 0.5), Span(99.9, 120.0)]
    result = normalize_intervals(spans, 100.0)
    assert result[0] == Span(0.0, 0.5)
    assert result[1] == Span(1.0, 3.0)  # 40 ms gap merged
    assert round(result[2].duration, 9) == 0.06  # lengthened around 5.005
    assert result[3] == Span(99.9, 100.0)


def test_command_file_rearms_afade_around_each_span() -> None:
    text, initial = command_file([Span(1.0, 2.0), Span(4.0, 5.0)], "mute0", 0.01)
    assert text.splitlines() == [
        "0.000000-1.000000 [enter] afade@mute0 t out, [enter] afade@mute0 st 1.000000, [enter] afade@mute0 d 0.010000;",
        "1.010000-1.990000 [enter] afade@mute0 t in, [enter] afade@mute0 st 1.990000, [enter] afade@mute0 d 0.010000;",
        "2.000000-4.000000 [enter] afade@mute0 t out, [enter] afade@mute0 st 4.000000, [enter] afade@mute0 d 0.010000;",
        "4.010000-4.990000 [enter] afade@mute0 t in, [enter] afade@mute0 st 4.990000, [enter] afade@mute0 d 0.010000;",
    ]
    assert initial == "t=in:ss=0:ns=1:curve=qsin"


def test_span_at_zero_starts_faded_out_and_short_spans_shrink_the_fade() -> None:
    text, initial = command_file([Span(0.0, 0.03)], "mute0", 0.01)
    assert initial == "t=out:ss=0:d=0.010000:curve=qsin"
    assert text.splitlines() == [
        "0.010000-0.020000 [enter] afade@mute0 t in, [enter] afade@mute0 st 0.020000, [enter] afade@mute0 d 0.010000;"
    ]
    text, _ = command_file([Span(1.0, 1.015)], "mute0", 0.01)
    assert "d 0.005000;" in text


@pytest.mark.parametrize(
    ("codec", "suffix", "extra", "expected"),
    [
        ("aac", ".mp4", {"bit_rate": 192_000}, ("aac", "192k")),
        ("aac", ".mkv", {"channels": 6}, ("aac", "384k")),
        ("ac3", ".mkv", {"bit_rate": 900_000, "channels": 6}, ("ac3", "640k")),
        ("dts", ".mkv", {}, ("flac", None)),
        ("truehd", ".mp4", {}, ("aac", "128k")),
        ("dts", ".ts", {"channels": 6}, ("ac3", "640k")),
        ("pcm_s16le", ".mov", {}, ("pcm_s16le", None)),
        ("vorbis", ".webm", {}, ("libopus", "96k")),
    ],
)
def test_encoder_choice(
    codec: str, suffix: str, extra: dict[str, Any], expected: tuple[str, str | None]
) -> None:
    assert choose_encoder(stream(1, codec=codec, **extra), suffix, ENCODERS, "auto", "auto") == expected


def test_explicit_codec_and_bitrate() -> None:
    assert choose_encoder(stream(1, codec="aac"), ".mkv", ENCODERS, "flac", "auto") == ("flac", None)
    assert choose_encoder(stream(1, codec="aac"), ".mkv", ENCODERS, "auto", "256k") == ("aac", "256k")
    with pytest.raises(DependencyError, match="libfdk_aac"):
        choose_encoder(stream(1, codec="aac"), ".mkv", ENCODERS, "libfdk_aac", "auto")


def test_webm_without_vorbis_or_opus_is_an_error() -> None:
    with pytest.raises(DependencyError, match=r"\.webm"):
        choose_encoder(stream(1, codec="vorbis"), ".webm", frozenset({"aac"}), "auto", "auto")


def test_disposition_value() -> None:
    assert disposition_value(stream(1, disposition={"default": 1, "forced": 0})) == "default"
    assert (
        disposition_value(stream(1, disposition={"default": 1, "hearing_impaired": 1}))
        == "default+hearing_impaired"
    )
    assert disposition_value(stream(1)) == "0"


def plan(output: OutputConfig, *streams: StreamInfo) -> dict[int, str]:
    media = info(stream(0, "video"), *streams)
    result = plan_streams(media, streams[0], output, "en")
    return {a.stream.index: a.action for a in result.actions}


def test_other_audio_auto_censors_same_dialogue_and_drops_the_rest() -> None:
    actions = plan(
        OutputConfig(),
        stream(1, codec="eac3", language="eng"),
        stream(2, codec="aac", language="eng"),
        stream(3, codec="aac", language="eng", title="Commentary"),
        stream(4, codec="aac", language="fra"),
        stream(5, codec="aac", language="eng", disposition={"visual_impaired": 1}),
    )
    assert actions == {0: "copy", 1: "censor", 2: "censor"}


@pytest.mark.parametrize(("policy", "expected"), [("copy", "copy"), ("censor", "censor"), ("drop", None)])
def test_other_audio_policies(policy: str, expected: str | None) -> None:
    actions = plan(
        OutputConfig(other_audio_streams=policy), stream(1, language="eng"), stream(2, language="fra")
    )  # type: ignore[arg-type]
    assert actions.get(2) == expected


def test_text_subtitles_are_censored_image_ones_copied_and_data_streams_dropped() -> None:
    streams = (stream(1, language="eng"), stream(2, "subtitle", codec="subrip"), stream(3, "data"),
               stream(4, "attachment", codec="ttf"), stream(5, "subtitle", codec="hdmv_pgs_subtitle"))  # fmt: skip
    assert plan(OutputConfig(), *streams) == {0: "copy", 1: "censor", 2: "censor", 4: "copy", 5: "copy"}
    assert plan(OutputConfig(subtitle_streams="copy"), *streams) == {
        0: "copy",
        1: "censor",
        2: "copy",
        4: "copy",
        5: "copy",
    }
    assert plan(OutputConfig(subtitle_streams="drop"), *streams) == {0: "copy", 1: "censor", 4: "copy"}


def test_subtitle_notes_name_image_streams_and_other_languages() -> None:
    media = info(stream(1, language="eng"), stream(2, "subtitle", codec="subrip", language="fre"),
                 stream(3, "subtitle", codec="subrip", language="eng"), stream(4, "subtitle", codec="dvd_subtitle"))  # fmt: skip
    notes = plan_streams(media, media.streams[0], OutputConfig(), "en").notes
    assert notes == (
        "subtitle stream #2 subrip fre is masked with the 'en' word list, which does not cover its language",
        "subtitle stream #4 dvd_subtitle copied uncensored: image-based subtitles cannot be edited",
    )
    dropped = plan_streams(media, media.streams[0], OutputConfig(), "en").drop(2, "could not extract it")
    assert [a.stream.index for a in dropped.actions] == [1, 3, 4] and dropped.notes[
        -1
    ] == "could not extract it"


def test_censored_subtitles_are_read_from_their_own_files_with_their_tags() -> None:
    media = info(
        stream(0, "video"),
        stream(1, codec="aac", language="eng"),
        stream(2, "subtitle", codec="subrip", language="eng", title="SDH", disposition={"default": 1}),
        stream(3, "subtitle", codec="hdmv_pgs_subtitle", language="eng"),
        stream(4, "subtitle", codec="ass", language="eng"),
    )
    streams_plan = plan_streams(media, media.streams[1], OutputConfig(), "en")
    files = {2: Path("/work/censored-2.srt"), 4: Path("/work/censored-4.ass")}
    command = build_command(stub_ffmpeg(), media, streams_plan, [], fade=0.01, output=OutputConfig(),
                            target=Path("out.mkv"), subtitle_files=files, tag="0.1.0;abc")  # fmt: skip
    args = command.args
    assert [args[i + 1] for i, a in enumerate(args) if a == "-i"] == [
        "file:" + str(Path("movie.mkv").resolve()), "file:" + str(files[2].resolve()), "file:" + str(files[4].resolve())
    ]  # fmt: skip
    assert [args[i + 1] for i, a in enumerate(args) if a == "-map"] == ["0:0", "0:1", "1:0", "0:3", "2:0"]
    assert args[args.index("-c:s:0") + 1] == "srt" and args[args.index("-c:s:2") + 1] == "ass"
    assert "-c:s:1" not in args  # the image-based stream is copied
    assert args[args.index("-metadata:s:s:0") + 1] == "language=eng"
    assert "title=SDH" in args and args[args.index("-disposition:s:0") + 1] == "default"
    assert args[args.index("-disposition:s:2") + 1] == "0"
    assert args[args.index("-metadata") + 1] == "VBR_CENSORED=0.1.0;abc"
    mp4 = build_command(stub_ffmpeg(), info(*media.streams[:3], name="movie.mp4"), plan_streams(
        info(*media.streams[:3], name="movie.mp4"), media.streams[1], OutputConfig(), "en"), [], fade=0.01,
        output=OutputConfig(), target=Path("out.mp4"), subtitle_files=files, tag="0.1.0;abc")  # fmt: skip
    assert mp4.args[mp4.args.index("-c:s:0") + 1] == "mov_text"
    assert mp4.args[mp4.args.index("-movflags") + 1] == "+faststart+use_metadata_tags"


def test_build_command_maps_in_order_and_reapplies_stream_tags() -> None:
    media = info(
        stream(0, "video"),
        stream(1, codec="eac3", language="eng", title="English 5.1", sample_rate=48_000, channels=6,
               disposition={"default": 1}),
        stream(2, "subtitle", codec="subrip"),
        name="movie.mp4",
    )  # fmt: skip
    streams_plan = plan_streams(media, media.streams[1], OutputConfig(subtitle_streams="copy"), "en")
    command = build_command(
        stub_ffmpeg(),
        media,
        streams_plan,
        [Span(1.0, 2.0)],
        fade=0.01,
        output=OutputConfig(),
        target=Path("out.mp4"),
    )
    args = command.args
    assert args[args.index("-filter_complex_script") + 1] == "graph.txt"
    maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
    assert maps == ["0:0", "[outmute0]", "0:2"]
    assert args.index("-c") < args.index("-c:a:0")
    assert args[args.index("-c:a:0") + 1] == "eac3"
    assert "language=eng" in args and "title=English 5.1" in args
    assert args[args.index("-disposition:a:0") + 1] == "default"
    assert "+faststart" in args
    assert command.files["graph.txt"] == (
        "[0:1]asetnsamples=n=480:p=0,asendcmd=f=mute0.cmd,afade@mute0=t=in:ss=0:ns=1:curve=qsin[outmute0]\n"
    )
    assert "mute0.cmd" in command.files
    assert command.censored_positions == [0]


def test_ffmpeg_7_loads_the_graph_with_the_slash_syntax() -> None:
    media = info(stream(1, codec="aac", language="eng"))
    streams_plan = plan_streams(media, media.streams[0], OutputConfig(), "en")
    command = build_command(stub_ffmpeg("ffmpeg version 7.1"), media, streams_plan, [Span(1.0, 2.0)],
                            fade=0.01, output=OutputConfig(), target=Path("out.mkv"))  # fmt: skip
    assert "-/filter_complex" in command.args


def test_without_spans_everything_is_copied() -> None:
    media = info(stream(0, "video"), stream(1, codec="aac", language="eng"))
    streams_plan = plan_streams(media, media.streams[1], OutputConfig(), "en")
    command = build_command(stub_ffmpeg(), media, streams_plan, [], fade=0.01, output=OutputConfig(),
                            target=Path("out.mkv"))  # fmt: skip
    assert command.files == {} and command.censored_positions == []
    assert "-filter_complex_script" not in command.args
    assert [command.args[i + 1] for i, a in enumerate(command.args) if a == "-map"] == ["0:0", "0:1"]
