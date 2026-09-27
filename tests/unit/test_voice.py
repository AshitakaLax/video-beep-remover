"""Voice replacement's choices and array helpers (DESIGN.md §16, §17.6), without its models."""

import sys
from typing import Any

import numpy as np
import pytest

from helpers import say
from video_beep_remover.context.analyse import ContextResult, Verdict, choose_substitutes
from video_beep_remover.context.lines import Line
from video_beep_remover.errors import DependencyError
from video_beep_remover.models import CensorInterval, Detection
from video_beep_remover.report import review_srt
from video_beep_remover.voice import _said
from video_beep_remover.voice.models import check_installed
from video_beep_remover.voice.splice import change, dialogue_channels, fade_mask, fit_case, utterance

TABLE = {"damn": ["darn"], "*fuck*": ["freaking", "frick"], "hell": ["heck"]}


def detection(word: str, start: float, end: float, term: str | None = None) -> Detection:
    return Detection(start, end, f" {word}", term or word.lower(), "mild", 0.9, "asr", None)


def verdict(use: str = "profane", reason: str = "listed", delivery: str | None = None) -> Verdict:
    return Verdict(use, reason, 0, {}, None, delivery, "low", False)  # type: ignore[arg-type]


class Judge:
    name = "fake-judge"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.questions: list[str] = []

    def ask(self, prompt: str) -> str:
        self.questions.append(prompt)
        return self.answer


def choose(verdicts: list[Verdict], found: list[Detection], judge: Judge | None = None) -> list[Any]:
    result = ContextResult(verdicts, [], [Line(0.0, 3.0, "Get the fuck out, damn it.", cue=1)])
    return [(c.substitute, c.reason) for c in choose_substitutes(found, result, TABLE, judge)]


def test_a_word_is_replaced_only_when_the_rules_allow() -> None:
    found = [detection("damn", 1.0, 1.3)] * 5 + [detection("crap", 1.0, 1.3)]
    verdicts = [
        verdict(),
        verdict(use="unsure", reason="not judged"),
        verdict(reason="sexual line"),
        verdict(delivery="shouted"),
        verdict(delivery="whispered"),
        verdict(),
    ]
    assert choose(verdicts, found) == [
        ("darn", "the only substitute"),
        (None, "the use is unsure"),
        (None, "a sexual line"),
        (None, "shouted delivery"),
        (None, "whispered delivery"),
        (None, "no substitute for this term"),
    ]


def test_the_judge_picks_among_several_substitutes() -> None:
    fucking = detection("fucking", 1.0, 1.3, term="*fuck*")
    assert choose([verdict()], [fucking]) == [(None, "several substitutes, and no judge to choose")]
    judge = Judge('{"substitute": "Freaking"}')
    assert choose([verdict()], [fucking], judge) == [("freaking", "judge")]
    assert '"freaking", "frick"' in judge.questions[0] and '>> "Get the fuck out' in judge.questions[0]
    for answer in ('{"substitute": null}', '{"substitute": "fudge"}', "no idea"):  # none, or not offered
        assert choose([verdict()], [fucking], Judge(answer)) == [(None, "no substitute fits, the judge says")]


def test_the_sentence_is_said_again_with_the_substitute() -> None:
    heard = say("We are late. What the hell is going on?", 1.0, 3.4)
    hell = heard[5]
    spoken = utterance(heard, detection("hell", hell.start, hell.end), "heck")
    assert spoken is not None
    assert (spoken.original, spoken.text) == ("What the hell is going on?", "What the heck is going on?")
    assert spoken.start == heard[3].start and spoken.end == heard[-1].end
    phrase = say("You son of a bitch!", 0.0, 1.5)
    whole = detection("son of a bitch!", phrase[1].start, phrase[4].end, term="son of a bitch")
    assert utterance(phrase, whole, "son of a gun") is not None
    assert utterance(phrase, whole, "son of a gun").text == "You son of a gun!"  # type: ignore[union-attr]
    assert utterance(heard, detection("hell", 9.0, 9.3), "heck") is None  # nothing heard there


def test_the_substitute_takes_the_case_and_punctuation_of_the_word() -> None:
    assert [fit_case(w, "heck") for w in ("hell", "Hell,", "HELL!", "(hell")] == [
        "heck",
        "Heck,",
        "HECK!",
        "(heck",
    ]


def test_dialogue_is_edited_in_the_centre_channel_of_surround_sound() -> None:
    assert dialogue_channels("5.1(side)", 6) == [2]
    assert dialogue_channels("7.1", 8) == [2]
    assert dialogue_channels("stereo", 2) == [0, 1]
    assert dialogue_channels("quad", 4) == [0, 1]  # no centre channel
    assert dialogue_channels("mono", 1) == [0]


def test_the_change_is_confined_to_the_span_and_fades_in_and_out() -> None:
    rate = 1000
    mask = fade_mask(1000, rate, 0.4, 0.6, fade=0.02)
    assert mask[:400].sum() == 0 and mask[600:].sum() == 0 and np.all(mask[420:580] == 1)
    assert 0 < mask[410] < 1 and 0 < mask[590] < 1

    window = np.zeros((6, 1000), dtype=np.float32)
    window[2] = 0.5  # the voice, in the centre channel
    window[0] = 0.3  # music, left
    vocals = window[[2]]
    edited = np.full(1000, 0.1, dtype=np.float32)
    delta = change(window, vocals, edited, (0.4, 0.6), rate, [2])
    assert not delta[[0, 1, 3, 4, 5]].any()  # only the dialogue channel changes
    assert np.allclose((window + delta)[2, 420:580], 0.1)  # the new voice, inside the span
    assert not delta[2, :400].any() and not delta[2, 600:].any()


def test_the_substitute_must_be_heard() -> None:
    assert _said("heck", say("What the heck", 0, 1))
    assert _said("son of a gun", say("son of a gun", 0, 1))
    assert _said("fricking", say("frickin'", 0, 1))  # close enough
    assert not _said("heck", say("What the hell", 0, 1))
    assert not _said("son of a gun", say("son of a bitch", 0, 1))


def test_review_subtitles_show_replaced_words() -> None:
    text = review_srt(
        [CensorInterval(0.9, 1.4), CensorInterval(2.0, 2.5)],
        [detection("damn", 1.0, 1.3), detection("hell", 2.1, 2.4)],
        replaced=[(0.9, 1.4, "damn → darn")],
    )
    assert "[replaced] damn → darn" in text and "[muted] hell" in text


def test_without_the_extra_replacement_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "f5_tts", None)
    with pytest.raises(DependencyError, match=r"video-beep-remover\[voice\]"):
        check_installed()
