from pathlib import Path

import pytest

from helpers import srt
from video_beep_remover.config.schema import Category, Hints, LexiconConfig
from video_beep_remover.detect.lexicon import Lexicon, compile_lexicon
from video_beep_remover.subtitles.censor import (
    censor_subtitles,
    censored_copy_path,
    mask_text,
    subtitle_format,
)
from video_beep_remover.subtitles.output import write_censored_copy

LEXICON: Lexicon = compile_lexicon(
    LexiconConfig(
        allow=["bastardiz*"],
        categories={
            "strong": Category(terms=["*fuck*", "*shit*", "bitch*", "son of a bitch", "bastard*"]),
            "mild": Category(terms=["hell", "damn"]),
            "religious": Category(terms=["oh my [god]"]),
        },
        hints=Hints(terms=["freak*", "heck"]),
    )
)


@pytest.mark.parametrize(
    ("text", "first_letter", "asterisks", "removed"),
    [
        ("What the hell?", "What the h***?", "What the ****?", "What the?"),
        ("Fuck you!", "F*** you!", "**** you!", "you!"),
        ("This is bullshit.", "This is b*******.", "This is ********.", "This is."),
        ("You son of a bitch", "You s** o* a b****", "You *** ** * *****", "You"),
        ("Oh my God, it's good", "Oh my G**, it's good", "Oh my ***, it's good", "Oh my, it's good"),
        ("Who said f***?", "Who said f***?", "Who said ****?", "Who said?"),
        ("Motherfucking hell--no", "M************ h***--no", "************* ****--no", "--no"),
        (
            "It's heck, freaking heck",
            "It's heck, freaking heck",
            "It's heck, freaking heck",
            "It's heck, freaking heck",
        ),
        ("A bastardized bastard", "A bastardized b******", "A bastardized *******", "A bastardized"),
    ],
)
def test_masks(text: str, first_letter: str, asterisks: str, removed: str) -> None:
    assert mask_text(text, LEXICON, "first_letter").text == first_letter
    assert mask_text(text, LEXICON, "asterisks").text == asterisks
    assert mask_text(text, LEXICON, "remove").text == removed


def test_markup_is_kept_and_takes_no_space() -> None:
    done = mask_text('<i>What the hell</i> is <font color="#ff0">that</font>?', LEXICON, "first_letter")
    assert done.text == '<i>What the h***</i> is <font color="#ff0">that</font>?'
    assert done.masked == 1
    # a tag inside a word does not split it; a line break does, and phrases run across it
    assert (
        mask_text("{\\an8}F{\\i1}uck{\\i0} it", LEXICON, "first_letter").text == "{\\an8}F{\\i1}***{\\i0} it"
    )
    assert mask_text("Son of a\\Nbitch", LEXICON, "asterisks").text == "*** ** *\\N*****"
    assert mask_text("hell\\Nno", LEXICON, "remove").text == "\\Nno"  # the line break stays
    assert mask_text("<v Joe>Damn</v>", LEXICON, "asterisks").text == "<v Joe>****</v>"


def test_nothing_to_mask_returns_the_text_unchanged() -> None:
    done = mask_text("A perfectly nice line", LEXICON, "first_letter")
    assert (done.text, done.masked) == ("A perfectly nice line", 0)


def test_srt_cues_are_masked_and_everything_else_kept() -> None:
    text = (
        "1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>What the hell</i> is that?\r\n- Son of a\r\nbitch!\r\n\r\n"
        "2\r\n00:00:03,000 --> 00:00:04,000\r\n{\\an8}Damn\r\n\r\n"
    )
    done = censor_subtitles(text, LEXICON, "first_letter")
    assert done.text == (
        "1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>What the h***</i> is that?\r\n- S** o* a\r\nb****!\r\n\r\n"
        "2\r\n00:00:03,000 --> 00:00:04,000\r\n{\\an8}D***\r\n\r\n"
    )
    assert done.masked == 6


def test_webvtt_headers_notes_and_settings_are_kept() -> None:
    text = (
        "WEBVTT - hell\n\nNOTE damn this\n\nintro\n00:01.000 --> 00:02.000 align:start\n"
        "<c.yellow>Damn</c> it\n\n00:03.000 --> 00:04.000\nFine\n"
    )
    done = censor_subtitles(text, LEXICON, "asterisks")
    assert done.text == text.replace("<c.yellow>Damn</c>", "<c.yellow>****</c>")
    assert subtitle_format(text) == "vtt"


def test_ass_dialogue_text_is_masked_but_not_styles_or_comments() -> None:
    text = (
        "[Script Info]\nTitle: hell\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize\nStyle: Hell,Arial,16\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Comment: 0,0:00:00.00,0:00:01.00,Hell,,0,0,0,,damn\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Hell,Bob,0,0,0,,{\\i1}Damn{\\i0}, hell, and more\\Nshit\n"
    )
    done = censor_subtitles(text, LEXICON, "first_letter")
    assert done.text == text.replace(
        "{\\i1}Damn{\\i0}, hell, and more\\Nshit", "{\\i1}D***{\\i0}, h***, and more\\Ns***"
    )
    assert done.masked == 3
    assert subtitle_format(text) == "ass"


def test_other_formats_go_through_pysubs2() -> None:
    done = censor_subtitles("{25}{50}What the hell|Oh damn\n", LEXICON, "asterisks", fps=25.0)
    assert done.masked == 2
    assert "****" in done.text and "hell" not in done.text


def test_srt_helper_output_round_trips() -> None:
    text = srt((1.0, 2.0, "Damn it"), (3.0, 4.0, "Nothing here"))
    done = censor_subtitles(text, LEXICON, "first_letter")
    assert done.text == text.replace("Damn", "D***") and done.masked == 1


def test_censored_copy_is_named_after_the_output(tmp_path: Path) -> None:
    video = tmp_path / "Movie.mkv"
    output = tmp_path / "out" / "Movie.clean.mkv"
    assert censored_copy_path(tmp_path / "Movie.en.sdh.SRT", video, output, "en") == (
        tmp_path / "out" / "Movie.clean.en.sdh.srt"
    )
    assert (
        censored_copy_path(tmp_path / "Movie.srt", video, output, None)
        == tmp_path / "out" / "Movie.clean.srt"
    )
    # a file in Subs/ not named after the video: the language, if known, names it
    assert censored_copy_path(tmp_path / "Subs" / "2_English.srt", video, output, "en") == (
        tmp_path / "out" / "Movie.clean.en.srt"
    )
    assert (
        censored_copy_path(tmp_path / "Movies.srt", video, output, None)
        == tmp_path / "out" / "Movie.clean.srt"
    )


@pytest.mark.parametrize("backup", [None, "Movie.orig.mkv"])
def test_in_place_the_censored_copy_takes_the_subtitle_files_place(
    tmp_path: Path, backup: str | None
) -> None:
    video = tmp_path / "Movie.mkv"
    subtitle = tmp_path / "Movie.en.srt"
    subtitle.write_text(srt((1.0, 2.0, "Damn it")), "utf-8")
    copy = write_censored_copy(
        subtitle, video=video, output=video, language="en", lexicon=LEXICON, mask="first_letter",
        fps=None, overwrite=False, backup=tmp_path / backup if backup else None,
    )  # fmt: skip
    assert copy.path == subtitle and "D*** it" in subtitle.read_text("utf-8")
    if backup:  # --backup: the unmodified file is kept next to the video's backup
        assert "Damn it" in (tmp_path / "Movie.orig.en.srt").read_text("utf-8")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["Movie.en.srt"] + (["Movie.orig.en.srt"] if backup else [])
    )


def test_removing_keeps_line_breaks_and_drops_what_is_left_empty() -> None:
    assert mask_text("You son of a\nbitch!", LEXICON, "remove").text == "You\n!"
    text = srt(
        (1.0, 2.0, "Son of a\nbitch\nI hate you"),
        (3.0, 4.0, "Bullshit"),  # nothing left: the cue goes
        (5.0, 6.0, "<i>Fuck</i>\nFine"),  # a line left with markup only goes too
    )
    done = censor_subtitles(text, LEXICON, "remove")
    assert done.text == (
        "1\n00:00:01,000 --> 00:00:02,000\nI hate you\n\n3\n00:00:05,000 --> 00:00:06,000\nFine\n\n"
    )
    assert done.masked == 6
    vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\nDamn\n\n00:03.000 --> 00:04.000\nFine\n"
    assert censor_subtitles(vtt, LEXICON, "remove").text == "WEBVTT\n\n00:03.000 --> 00:04.000\nFine\n"
    # The other masks never empty a line.
    assert censor_subtitles(text, LEXICON, "asterisks").text.count("\n") == text.count("\n")
