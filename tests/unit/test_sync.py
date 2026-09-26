import random
from collections.abc import Callable, Sequence

import pytest

from helpers import say
from video_beep_remover.config.schema import SyncConfig
from video_beep_remover.models import Cue, SyncModel, Word
from video_beep_remover.subtitles.sync import best_match, check_sync, choose_anchors, cue_tokens, fit, snap

VOCABULARY = [
    f"{a}{b}"
    for a in ("har", "lan", "fal", "mea", "cop", "vio", "gla", "orc", "pil", "thu", "mar", "sad")
    for b in ("bor", "tern", "con", "dow", "per", "lin", "cier", "chard", "grim", "der", "ble", "dle")
]


def film(count: int = 120, length: float = 7200.0) -> list[Cue]:
    """A long film's worth of cues. Lines are random words, so no two lines look alike."""
    rng = random.Random(7)
    cues = []
    for i in range(count):
        start = 10 + i * (length - 20) / count
        cues.append(Cue(i + 1, start, start + 3.0, " ".join(rng.sample(VOCABULARY, 6))))
    return cues


def speaker(
    cues: Sequence[Cue], model: SyncModel, *, paraphrase: bool = False, edit: bool = False
) -> tuple[list[tuple[float, float]], Callable[[float, float], list[Word]]]:
    """What the audio says: each cue spoken at model.to_media(cue time), 0.2 s after the cue starts.

    `paraphrase` says something else entirely; `edit` changes one word in six."""
    spoken: list[Word] = []
    for cue in cues:
        text = cue.text
        if paraphrase:
            text = "Honestly I think it was over there somewhere"
        elif edit:
            text = text.rsplit(" ", 1)[0] + " today"
        spoken += say(text, model.to_media(cue.start) + 0.2, model.to_media(cue.end) - 0.3)
    calls: list[tuple[float, float]] = []

    def transcribe(start: float, end: float) -> list[Word]:
        calls.append((start, end))
        return [w for w in spoken if start <= w.start and w.end <= end]

    return calls, transcribe


def test_anchors_are_spread_and_prefer_rare_words() -> None:
    cues = [
        Cue(1, 5.0, 7.0, "Yes sir right away sir"),
        Cue(2, 8.0, 10.0, "The copper lantern is broken again"),
        Cue(3, 50.0, 52.0, "Yes sir right away sir"),
        Cue(4, 55.0, 56.0, "No."),  # too short
        Cue(5, 60.0, 62.0, "Sing the falcon song with me", lyrics=True),
        Cue(6, 70.0, 72.0, "Yes sir right away sir"),
        Cue(7, 95.0, 97.0, "Meet me at the marble orchard"),
    ]
    assert [c.index for c in choose_anchors(cues, 2)] == [2, 7]
    assert [c.index for c in choose_anchors(cues, 3)] == [2, 3, 7]
    assert choose_anchors(cues, 0) == []


def test_short_cues_are_used_when_nothing_else_qualifies() -> None:
    cues = [Cue(1, 1.0, 1.8, "Get down!"), Cue(2, 5.0, 5.9, "Run, now!")]
    assert [c.index for c in choose_anchors(cues, 6)] == [1, 2]
    assert choose_anchors([Cue(1, 1.0, 1.2, "Oh.")], 6) == []


def test_best_match_finds_the_line_among_other_speech() -> None:
    words = say("so anyway we found the copper lantern number nine yesterday right", 10.0, 16.0)
    score, fidelity, heard_at = best_match(
        cue_tokens(Cue(1, 0, 1, "We found the copper lantern, number nine yesterday!")), words
    )
    assert score == 100 and fidelity == 1.0
    assert heard_at == pytest.approx(words[2].start)
    assert best_match(["hello"], [])[0] == 0.0


@pytest.mark.parametrize("ratio", [25 / 23.976, 23.976 / 25, 1.0])
def test_fit_recovers_frame_rate_ratios_and_offsets(ratio: float) -> None:
    rng = random.Random(4)
    xs = [100.0 * k for k in range(1, 12)]
    pairs = [(x, ratio * x - 2.5 + rng.uniform(-0.05, 0.05)) for x in xs]
    pairs[3] = (pairs[3][0], pairs[3][1] + 30.0)  # a mismatched anchor
    model = fit(pairs)
    assert model.scale == ratio  # snapped exactly
    assert model.offset == pytest.approx(-2.5, abs=0.05)
    assert model.error < 0.05


def test_fit_uses_an_offset_only_for_few_or_close_pairs() -> None:
    assert fit([(10.0, 11.0), (20.0, 21.2)]).scale == 1.0
    close = fit([(10.0, 11.0), (20.0, 21.2), (30.0, 31.1)])
    assert (close.scale, close.offset) == (1.0, pytest.approx(1.1))
    assert fit([]) == SyncModel()
    assert snap(1.0004) == 1.0 and snap(1.01) == 1.01


def test_trusted_subtitles_with_a_small_offset_pass() -> None:
    cues = film()
    calls, transcribe = speaker(cues, SyncModel(offset=1.2))
    result = check_sync(cues, duration=7300, transcribe=transcribe, config=SyncConfig(), trusted=True)
    assert result.passed, result.reason
    assert result.matched == len(result.anchors) == 6
    assert result.model.scale == 1.0
    assert result.model.offset == pytest.approx(1.2 + 0.2, abs=0.05)  # cues lead speech by 0.2 s here
    assert result.fidelity == 1.0
    assert all(end - start <= 3.0 + 6.0 + 1e-6 for start, end in calls)  # cue + ±3 s


def test_trusted_search_is_too_narrow_for_a_large_offset() -> None:
    cues = film()
    _, transcribe = speaker(cues, SyncModel(offset=10.0))
    result = check_sync(cues, duration=7300, transcribe=transcribe, config=SyncConfig(), trusted=True)
    assert not result.passed
    assert result.reason == "only 0 of 6 anchor cues were heard where the subtitles place them"
    untrusted = check_sync(cues, duration=7300, transcribe=transcribe, config=SyncConfig(), trusted=False)
    assert untrusted.passed and untrusted.model.offset == pytest.approx(10.2, abs=0.05)


def test_tracking_follows_a_small_drift() -> None:
    """An NTSC 0.1 % slowdown drifts 7 s over two hours: more than the ±3 s search, but each anchor
    is searched where the fit so far predicts it."""
    cues = film()
    truth = SyncModel(scale=30 / 29.97, offset=0.5)
    _, transcribe = speaker(cues, truth)
    result = check_sync(cues, duration=7300, transcribe=transcribe, config=SyncConfig(), trusted=True)
    assert result.passed, result.reason
    assert result.model.scale == 30 / 29.97
    assert result.model.offset == pytest.approx(0.7, abs=0.05)


def test_a_known_frame_rate_ratio_is_applied_up_front() -> None:
    """25 vs 23.976 fps drifts 5 minutes over two hours, which tracking alone cannot follow with
    anchors 20 minutes apart. Online subtitles state their frame rate, so the ratio is known."""
    cues = film()
    truth = SyncModel(scale=23.976 / 25, offset=3.0)
    _, transcribe = speaker(cues, truth)
    config = SyncConfig()
    blind = check_sync(cues, duration=7300, transcribe=transcribe, config=config, trusted=False)
    assert not blind.passed
    hinted = check_sync(
        cues, duration=7300, transcribe=transcribe, config=config, trusted=False, default_scale=23.976 / 25
    )
    assert hinted.passed, hinted.reason
    assert hinted.model.scale == 23.976 / 25


def test_paraphrased_subtitles_fail() -> None:
    cues = film()
    _, transcribe = speaker(cues, SyncModel(), paraphrase=True)
    result = check_sync(cues, duration=7300, transcribe=transcribe, config=SyncConfig(), trusted=True)
    assert not result.passed and result.reason is not None
    assert result.matched == 0 and result.fidelity is not None and result.fidelity < 0.6


def test_low_fidelity_fails_even_when_lines_match() -> None:
    cues = film()
    _, transcribe = speaker(cues, SyncModel(), edit=True)
    result = check_sync(
        cues, duration=7300, transcribe=transcribe, config=SyncConfig(min_fidelity=0.95), trusted=True
    )
    assert result.matched == 6
    assert result.fidelity is not None and 0.75 < result.fidelity < 0.95
    assert not result.passed and result.reason is not None and "not verbatim" in result.reason


def test_check_can_be_turned_off_and_needs_anchor_cues() -> None:
    cues = film()
    _, transcribe = speaker(cues, SyncModel())
    off = check_sync(cues, duration=7300, transcribe=transcribe, config=SyncConfig(anchors=0), trusted=True)
    assert off.passed and not off.checked and off.model == SyncModel()
    none = check_sync(
        [Cue(1, 1.0, 1.1, "Oh.")], duration=10, transcribe=transcribe, config=SyncConfig(), trusted=True
    )
    assert not none.passed and none.reason == "no cue is usable as a sync anchor"
