from helpers import words
from video_beep_remover.config.schema import Category, LexiconConfig
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.detect.matcher import Token, detect_in_words, find_matches
from video_beep_remover.models import Word


def lexicon(allow: tuple[str, ...] = (), **categories: list[str]):  # type: ignore[no-untyped-def]
    return compile_lexicon(
        LexiconConfig(
            allow=list(allow), categories={name: Category(terms=terms) for name, terms in categories.items()}
        )
    )


def tokens(text: str) -> list[Token]:
    return [Token.from_raw(raw) for raw in text.split()]


def test_longest_match_wins() -> None:
    matches = find_matches(lexicon(strong=["bitch*", "son of a bitch"]), tokens("you son of a bitch!"))
    assert [(m.term, m.start, m.end) for m in matches] == [("son of a bitch", 1, 5)]


def test_bracketed_target_limits_what_is_censored() -> None:
    matches = find_matches(lexicon(religious=["oh my [god]"]), tokens("Oh my God, that's good"))
    assert [(m.term, m.targets) for m in matches] == [("oh my [god]", (2,))]
    assert find_matches(lexicon(religious=["oh my [god]"]), tokens("my god")) == []


def test_allowlist_vetoes_single_word_matches() -> None:
    found = find_matches(lexicon(("bastardiz*",), strong=["bastard*"]), tokens("a bastardized bastard"))
    assert [m.start for m in found] == [2]


def test_first_category_wins_a_tie() -> None:
    matches = find_matches(lexicon(strong=["hell"], mild=["hell"]), tokens("hell"))
    assert [m.category for m in matches] == ["strong"]


def test_each_token_is_used_once() -> None:
    matches = find_matches(lexicon(strong=["*fuck*", "fucking"]), tokens("fucking"))
    assert len(matches) == 1


def test_detections_take_word_timings_and_lowest_confidence() -> None:
    spoken = [Word(" son", 1.0, 1.2, 0.9), Word(" of", 1.2, 1.3, 0.8), Word(" a", 1.3, 1.35, 0.95),
              Word(" bitch.", 1.35, 1.7, 0.6)]  # fmt: skip
    [detection] = detect_in_words(lexicon(strong=["son of a bitch"]), spoken)
    assert (detection.start, detection.end) == (1.0, 1.7)
    assert detection.heard == "son of a bitch."
    assert detection.confidence == 0.6
    assert detection.source == "asr"


def test_separate_targets_give_separate_detections() -> None:
    spoken = words(("son", 1.0, 1.2), ("of", 1.2, 1.3), ("a", 1.3, 1.35), ("bitch", 1.35, 1.7))
    found = detect_in_words(lexicon(strong=["[son] of a [bitch]"]), spoken)
    assert [(d.heard, d.start, d.end) for d in found] == [("son", 1.0, 1.2), ("bitch", 1.35, 1.7)]


def test_masked_words_from_whisper_are_detected() -> None:
    spoken = words(("what", 0.0, 0.2), ("the", 0.2, 0.3), ("f***", 0.3, 0.6))
    [detection] = detect_in_words(lexicon(strong=["*fuck*"]), spoken)
    assert detection.category == "strong"
    assert detection.term == "masked:f***"


def test_hyphenated_compound_matches_joined_term() -> None:
    [detection] = detect_in_words(lexicon(strong=["motherfucker"]), words(("mother-fucker", 2.0, 2.6)))
    assert detection.heard == "mother-fucker"


def test_punctuation_only_words_are_ignored() -> None:
    assert detect_in_words(lexicon(strong=["*fuck*"]), words(("...", 0.0, 0.1), ("-", 0.1, 0.2))) == []


def test_word_with_reversed_times_gets_zero_length() -> None:
    [detection] = detect_in_words(lexicon(mild=["damn"]), [Word("damn", 2.0, 1.9, 0.9)])
    assert detection.end == detection.start == 2.0
