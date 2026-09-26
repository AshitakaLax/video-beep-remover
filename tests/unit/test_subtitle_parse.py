from pathlib import Path

import pytest

from video_beep_remover.errors import SubtitleError
from video_beep_remover.models import Cue
from video_beep_remover.subtitles.parse import clean_text, decode_text, parse_subtitles, read_subtitle_file

SRT = """1
00:00:01,000 --> 00:00:03,500
<i>- JOHN: What the hell?</i>
- [door slams] Shit!

2
00:00:04,000 --> 00:00:05,000
♪ la la ♪

3
00:00:06,500 --> 00:00:08,000
(laughs)

4
00:00:06,000 --> 00:00:07,000
<font color="#ff0000">Red</font> {\\an8}top
"""


def test_srt_cues_are_cleaned_sorted_and_numbered() -> None:
    assert parse_subtitles(SRT) == [
        Cue(1, 1.0, 3.5, "What the hell? Shit!"),
        Cue(2, 4.0, 5.0, "la la", lyrics=True),
        Cue(3, 6.0, 7.0, "Red top"),
    ]


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("MAN 2: Get the f*** out!", "Get the f*** out!"),
        ("- Look: this is it.\n- No.", "Look: this is it. No."),  # only all-capital labels are removed
        ("10:30 is when we meet.", "10:30 is when we meet."),
        ("[man\nshouting] Hey!", "Hey!"),  # a description split over two lines
        ("[JOHN] - Hi.", "Hi."),
        ("# Happy birthday #", "Happy birthday"),
    ],
)
def test_clean_text(raw: str, spoken: str) -> None:
    assert clean_text(raw)[0] == spoken


def test_songs_are_marked_as_lyrics() -> None:
    assert clean_text("♪ Hit the road ♪") == ("Hit the road", True)
    assert clean_text("# Happy birthday #") == ("Happy birthday", True)
    assert clean_text("Room #5") == ("Room #5", False)


def test_webvtt_metadata_blocks_are_not_cue_text() -> None:
    vtt = (
        "WEBVTT\n\nNOTE written by\nsomeone\n\n00:01.000 --> 00:03.000\n<v Bob>Hello <b>there</b></v>\n\n"
        "NOTE a comment\n\n00:04.000 --> 00:05.000 align:start\n<c.yellow>Second</c> line <00:04.500>here\n"
    )
    assert [c.text for c in parse_subtitles(vtt)] == ["Hello there", "Second line here"]


def test_repeated_lines_in_overlapping_cues_become_one_cue() -> None:
    srt = (
        "1\n00:00:09,000 --> 00:00:10,000\nLook out!\n\n"
        "2\n00:00:10,000 --> 00:00:11,000\nLook out!\n\n"
        "3\n00:00:20,000 --> 00:00:21,000\nLook out!\n"
    )
    assert parse_subtitles(srt) == [Cue(1, 9.0, 11.0, "Look out!"), Cue(2, 20.0, 21.0, "Look out!")]


def test_microdvd_needs_the_frame_rate() -> None:
    text = "{25}{75}Hello there|General Kenobi\n"
    assert parse_subtitles(text, fps=25) == [Cue(1, 1.0, 3.0, "Hello there General Kenobi")]
    with pytest.raises(SubtitleError, match="Framerate"):
        parse_subtitles(text)


def test_garbage_is_a_subtitle_error() -> None:
    with pytest.raises(SubtitleError, match="not a subtitle file"):
        parse_subtitles("this is not a subtitle file")


def test_decoding_prefers_utf8_then_the_languages_code_page() -> None:
    text = "Café crème, naïve déjà vu ’ …"
    assert decode_text(text.encode("utf-8")) == text
    assert decode_text(text.encode("utf-8-sig")) == text
    assert decode_text(text.encode("utf-16")) == text
    assert decode_text(text.encode("cp1252"), "en") == text
    assert decode_text("Zażółć gęślą jaźń".encode("cp1250"), "pl") == "Zażółć gęślą jaźń"
    russian = "Привет, как дела? Это тестовая строка для проверки."
    assert decode_text(russian.encode("cp1251"), "ru") == russian
    assert decode_text(russian.encode("cp1251")) == russian  # detected without a hint


def test_read_subtitle_file_limits_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "movie.srt"
    text = "1\n00:00:01,000 --> 00:00:02,000\nCafé?\n"
    path.write_bytes(text.encode("cp1252"))
    assert read_subtitle_file(path, "en") == text
    monkeypatch.setattr("video_beep_remover.subtitles.parse.MAX_FILE_BYTES", 10)
    with pytest.raises(SubtitleError, match="too big"):
        read_subtitle_file(path)
    with pytest.raises(SubtitleError, match="could not read"):
        read_subtitle_file(tmp_path / "missing.srt")
