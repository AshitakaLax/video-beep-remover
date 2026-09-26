import pytest

from video_beep_remover.detect.normalize import collapse_runs, fold, normalize_token, token_forms


def test_fold_straightens_quotes_applies_nfkc_and_casefolds() -> None:
    assert fold("Christ’s") == "christ's"
    assert fold("ＤＡＭＮ") == "damn"  # full-width letters
    assert fold("STRASSE") == fold("straße")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hell!", "hell"),
        ("(damn)", "damn"),
        ('"Shit,"', "shit"),
        ("f***", "f***"),  # mask characters are kept
        ("***!", "***"),
        ("Fuuuuck?!", "fuck"),
        ("shiiit", "shit"),
        ("hell", "hell"),  # two identical letters stay
        ("1000", "1000"),  # digit runs stay
        ("fuckin'", "fuckin"),
        ("--", ""),
        ("♪", ""),
    ],
)
def test_normalize_token(raw: str, expected: str) -> None:
    assert normalize_token(raw) == expected


def test_collapse_runs_only_touches_letters() -> None:
    assert collapse_runs("brrr 999 aaah") == "br 999 ah"


def test_token_forms_adds_joined_form_for_hyphens() -> None:
    assert token_forms("mother-fucker") == ("mother-fucker", "motherfucker")
    assert token_forms("damn") == ("damn",)
    assert token_forms("") == ()
