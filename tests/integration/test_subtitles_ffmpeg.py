"""Embedded subtitle streams, extracted with real FFmpeg."""

from pathlib import Path

import pytest

from helpers import make_clip, run_ffmpeg
from video_beep_remover.errors import MediaError, SubtitleError
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe
from video_beep_remover.subtitles.acquire import SubtitleLoader, embedded_candidates
from video_beep_remover.subtitles.parse import parse_subtitles

pytestmark = pytest.mark.ffmpeg

SRT = "1\n00:00:01,000 --> 00:00:02,500\n<i>What the hell?</i>\n\n2\n00:00:03,000 --> 00:00:04,000\n- [sighs] Fine.\n"
ASS = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}Damn it{\\i0}, Jim.
Comment: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,not spoken
"""


@pytest.mark.parametrize("suffix", [".mkv", ".mp4"])
def test_embedded_text_streams_are_extracted_and_parsed(tmp_path: Path, suffix: str) -> None:
    source = make_clip(tmp_path / f"movie{suffix}", duration=5.0, subtitles=SRT)
    ff = FFmpeg()
    info = probe(ff, source)
    [candidate] = embedded_candidates(info)
    assert candidate.codec == ("subrip" if suffix == ".mkv" else "mov_text")
    text = SubtitleLoader(ff, source, tmp_path, [candidate]).text(candidate)
    cues = parse_subtitles(text)
    assert [c.text for c in cues] == ["What the hell?", "Fine."]
    # Within one AAC priming delay: the Matroska muxer shifts every stream to start at zero.
    assert [(c.start, c.end) for c in cues] == [
        (pytest.approx(1.0, abs=0.025), pytest.approx(2.5, abs=0.025)),
        (pytest.approx(3.0, abs=0.025), pytest.approx(4.0, abs=0.025)),
    ]


def test_several_streams_are_extracted_in_one_pass(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "plain.mkv", duration=3.0)
    ass = tmp_path / "styled.ass"
    ass.write_text(ASS, "utf-8")
    srt = tmp_path / "plain.srt"
    srt.write_text(SRT, "utf-8")
    source = tmp_path / "movie.mkv"
    run_ffmpeg("-i", str(clip), "-i", str(ass), "-i", str(srt), "-map", "0", "-map", "1", "-map", "2",
               "-c", "copy", "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=eng",
               str(source))  # fmt: skip
    ff = FFmpeg()
    candidates = embedded_candidates(probe(ff, source))
    assert [c.codec for c in candidates] == ["ass", "subrip"]

    runs = 0
    real_run = ff.run

    def counting_run(*args: object, **kwargs: object) -> None:
        nonlocal runs
        runs += 1
        real_run(*args, **kwargs)  # type: ignore[arg-type]

    ff.run = counting_run  # type: ignore[method-assign]
    loader = SubtitleLoader(ff, source, tmp_path, candidates)
    assert [c.text for c in parse_subtitles(loader.text(candidates[0]))] == ["Damn it, Jim."]
    assert [c.text for c in parse_subtitles(loader.text(candidates[1]))] == ["What the hell?", "Fine."]
    assert runs == 1


def test_a_failed_extraction_is_not_repeated(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "movie.mkv", duration=3.0, subtitles=SRT)
    ff = FFmpeg()
    [candidate] = embedded_candidates(probe(ff, source))
    runs = 0

    def failing_run(*args: object, **kwargs: object) -> None:
        nonlocal runs
        runs += 1
        raise MediaError("boom")

    ff.run = failing_run  # type: ignore[method-assign]
    loader = SubtitleLoader(ff, source, tmp_path, [candidate])
    for _ in range(2):
        with pytest.raises(SubtitleError):
            loader.text(candidate)
    assert runs == 1
