from pathlib import Path

import pytest

from video_beep_remover.config.schema import Category, Hints, LexiconConfig
from video_beep_remover.detect.lexicon import (
    MASKED_CATEGORY,
    TermError,
    compile_lexicon,
    parse_term,
    parse_word,
)
from video_beep_remover.errors import ConfigError


def lexicon_config(**categories: list[str]) -> LexiconConfig:
    return LexiconConfig(categories={name: Category(terms=terms) for name, terms in categories.items()})


@pytest.mark.parametrize(
    ("pattern", "matching", "not_matching"),
    [
        ("damn", ["damn"], ["damned", "goddamn"]),
        ("bitch*", ["bitch", "bitches", "bitchy"], ["sonofabitch"]),
        ("*shit", ["shit", "bullshit"], ["shitty"]),
        ("*shit*", ["shit", "bullshit", "shitty", "dipshit"], ["shiitake"]),
        ("christ's", ["christ's"], ["christs"]),
    ],
)
def test_word_patterns_match_whole_words(pattern: str, matching: list[str], not_matching: list[str]) -> None:
    word = parse_word(pattern)
    for text in matching:
        assert word.matches((text,)), text
    for text in not_matching:
        assert not word.matches((text,)), text


def test_hyphenated_pattern_also_matches_joined_form() -> None:
    word = parse_word("mother-fucker")
    assert word.matches(("motherfucker",))
    assert word.matches(("mother-fucker",))


def test_pattern_is_normalized_like_tokens() -> None:
    assert parse_word("DAMN").matches(("damn",))
    assert parse_word("Fuuuck").matches(("fuck",))


def test_bracketed_words_are_targets() -> None:
    term = parse_term("oh my [god]", "religious")
    assert [w.target for w in term.words] == [False, False, True]
    assert term.targets == (2,)
    assert parse_term("son of a bitch", "strong").targets == (0, 1, 2, 3)


def test_regex_terms_search_inside_the_word() -> None:
    term = parse_term("re:^f+u+c+k+", "strong")
    assert term.words[0].matches(("fucking",))
    assert not term.words[0].matches(("unfuck",))


@pytest.mark.parametrize("bad", ["", "*", "re:(unclosed"])
def test_invalid_terms_are_rejected(bad: str) -> None:
    with pytest.raises(TermError):
        parse_term(bad, "strong")


def test_only_enabled_categories_are_compiled_in_config_order() -> None:
    config = LexiconConfig(
        categories={
            "strong": Category(terms=["*fuck*"]),
            "mild": Category(enabled=False, terms=["damn"]),
            "religious": Category(terms=["goddamn*"]),
        }
    )
    lexicon = compile_lexicon(config)
    assert lexicon.categories == ("strong", "religious")
    assert [t.text for t in lexicon.terms] == ["*fuck*", "goddamn*"]


def test_only_enables_exactly_the_named_categories() -> None:
    config = LexiconConfig(
        categories={"strong": Category(terms=["*fuck*"]), "mild": Category(enabled=False, terms=["damn"])}
    )
    lexicon = compile_lexicon(config, only=["mild"])
    assert lexicon.categories == ("mild",)


def test_unknown_category_lists_available_ones() -> None:
    with pytest.raises(ConfigError, match=r"unknown categories: nope.*available: strong"):
        compile_lexicon(lexicon_config(strong=["x*y"]), only=["nope"])


def test_word_files_become_categories(tmp_path: Path) -> None:
    (tmp_path / "extra.txt").write_text("# my list\nfrick*\n\ncrikey  # comment\n", "utf-8")
    config = LexiconConfig(files=["extra.txt"])
    lexicon = compile_lexicon(config, base_dir=tmp_path)
    assert lexicon.categories == ("extra",)
    assert [t.text for t in lexicon.terms] == ["frick*", "crikey"]


def test_missing_word_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="word file not found"):
        compile_lexicon(LexiconConfig(files=["missing.txt"]), base_dir=tmp_path)


def test_allowlist_entries_must_be_single_words() -> None:
    with pytest.raises(ConfigError, match="single words"):
        compile_lexicon(LexiconConfig(allow=["hello world"]))


def test_broad_wildcards_produce_a_warning() -> None:
    lexicon = compile_lexicon(lexicon_config(strong=["as*", "ass*hole"]))
    assert any("'as*'" in warning for warning in lexicon.warnings)
    assert not any("ass*hole" in warning for warning in lexicon.warnings)


def test_bad_term_error_names_category_and_term() -> None:
    with pytest.raises(ConfigError, match=r"category 'strong', term 're:\('"):
        compile_lexicon(lexicon_config(strong=["re:("]))


def test_hints_are_compiled_separately() -> None:
    lexicon = compile_lexicon(LexiconConfig(hints=Hints(terms=["freak*"])))
    assert [t.text for t in lexicon.hints] == ["freak*"]
    assert lexicon.terms == ()


@pytest.mark.parametrize(
    ("token", "category"),
    [("d**n", "mild"), ("f***ing", "strong"), ("sh*t", "strong"), ("h***", "mild"), ("***", MASKED_CATEGORY)],
)
def test_masked_tokens_get_the_category_of_the_word_they_hide(token: str, category: str) -> None:
    lexicon = compile_lexicon(lexicon_config(strong=["*fuck*", "*shit*", "dickhead*"], mild=["damn", "hell"]))
    assert lexicon.is_masked(token)
    assert lexicon.masked_category(token) == category


def test_masked_token_of_a_disabled_category_is_left_alone() -> None:
    config = LexiconConfig(
        categories={"strong": Category(terms=["*fuck*"]), "mild": Category(enabled=False, terms=["damn"])}
    )
    lexicon = compile_lexicon(config)
    assert lexicon.masked_category("d**n") is None


@pytest.mark.parametrize("token", ["#1", "*", "a", "hello"])
def test_not_masked(token: str) -> None:
    assert not compile_lexicon(LexiconConfig()).is_masked(token)


def test_masked_detection_can_be_turned_off() -> None:
    lexicon = compile_lexicon(LexiconConfig(detect_masked=False))
    assert not lexicon.is_masked("f***")
