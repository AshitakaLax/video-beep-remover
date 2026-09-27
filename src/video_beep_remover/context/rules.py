"""The cheap signals of the context layer (DESIGN.md §17.3): sound descriptions, delivery, and
phrases matched like listed words."""

import re
from collections.abc import Sequence

from video_beep_remover.config.schema import Category, Hints, LexiconConfig
from video_beep_remover.context.lines import Line
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.detect.matcher import Token, find_matches
from video_beep_remover.detect.normalize import split_words

SEXUAL_SOUNDS = re.compile(
    r"\b(moan\w*|kiss\w*|sexual\w*|orgasm\w*|climax\w*|pant(s|ing)|breath(es|ing) heavily|"
    r"bed ?springs|bed creak\w*|making love|having sex|sex)\b"
)
_SHOUTED = re.compile(r"\b(shout\w*|yell\w*|scream\w*|screech\w*|bellow\w*|roar\w*)\b")
_WHISPERED = re.compile(r"\b(whisper\w*|murmur\w*|mutter\w*|under (his|her|their|my) breath)\b")
_TEARFUL = re.compile(r"\b(sob\w*|cry|cries|crying|weep\w*|tearful\w*)\b")
_LETTERS = re.compile(r"[^\W\d_]")


def sexual_sounds(line: Line) -> list[str]:
    return [sound for sound in line.sounds if SEXUAL_SOUNDS.search(sound)]


def _shouted_text(text: str) -> bool:
    letters = _LETTERS.findall(text)
    return len(letters) >= 4 and all(ch.isupper() for ch in letters)


def delivery(line: Line) -> str | None:
    """ "shouted", "whispered" or "tearful", from sound descriptions or an all-capitals line."""
    for pattern, name in ((_SHOUTED, "shouted"), (_WHISPERED, "whispered"), (_TEARFUL, "tearful")):
        if any(pattern.search(sound) for sound in line.sounds):
            return name
    return "shouted" if _shouted_text(line.text) else None


def intensity(line: Line) -> str:
    """ "high", "medium" or "low": shouting or repeated exclamation marks, one exclamation mark, or
    neither."""
    if delivery(line) == "shouted" or line.text.count("!") >= 2:
        return "high"
    return "medium" if "!" in line.text else "low"


class Phrases:
    """Terms in word-list syntax (§4.3), found in a line's text the way listed words are."""

    def __init__(self, terms: Sequence[str], name: str) -> None:
        self.lexicon = compile_lexicon(
            LexiconConfig(
                categories={name: Category(terms=list(terms))},
                detect_masked=False,
                hints=Hints(terms=[]),
            )
        )

    def find(self, text: str) -> list[str]:
        """The terms found in `text`, each once, in the order they occur."""
        if not text or not self.lexicon.terms:
            return []
        tokens = [Token.from_raw(raw) for raw, _, _ in split_words(text)]
        return list(dict.fromkeys(match.term for match in find_matches(self.lexicon, tokens)))
