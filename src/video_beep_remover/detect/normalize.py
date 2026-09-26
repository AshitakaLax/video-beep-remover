"""Text normalization shared by word-list terms and recognized words (DESIGN.md §4.3, rule 1)."""

import re
import unicodedata

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "´": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
    }
)
# Stripped from both ends of a token. Mask characters such as * and # are kept on purpose.
EDGE_PUNCTUATION = "\"'.,!?;:()[]{}<>«»¿¡…—–-_/\\|~^♪"
_LETTER_RUN = re.compile(r"([^\W\d_])\1{2,}")


def fold(text: str) -> str:
    """NFKC, straight quotes and case folding."""
    return unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()


def collapse_runs(text: str) -> str:
    """Reduce runs of three or more identical letters to one ("fuuuck" -> "fuck")."""
    return _LETTER_RUN.sub(r"\1", text)


def normalize_token(raw: str) -> str:
    """Normalize one word: fold, strip surrounding punctuation, collapse letter runs."""
    return collapse_runs(fold(raw).strip().strip(EDGE_PUNCTUATION).strip())


def token_forms(normalized: str) -> tuple[str, ...]:
    """The token plus its joined form when it contains hyphens ("mother-fucker" -> "motherfucker")."""
    if not normalized:
        return ()
    if "-" in normalized:
        joined = normalized.replace("-", "")
        if joined:
            return (normalized, joined)
    return (normalized,)


_SEPARATORS = re.compile(r"--+|\.\.\.+|[—–…]")  # dashes and ellipses between words, e.g. "hell--no"


def split_words(text: str) -> list[tuple[str, int, int]]:
    """Whitespace-separated words of subtitle text with their character spans; dashes and ellipses
    between words also separate them."""
    spaced = _SEPARATORS.sub(lambda m: " " * len(m.group()), text)
    return [(text[m.start() : m.end()], m.start(), m.end()) for m in re.finditer(r"\S+", spaced)]
