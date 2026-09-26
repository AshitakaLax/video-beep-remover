"""Find listed words in a sequence of tokens (DESIGN.md §4.3 rules 2-7, §6.9)."""

from collections.abc import Sequence
from dataclasses import dataclass

from video_beep_remover.detect.lexicon import MASKED_CATEGORY, Lexicon
from video_beep_remover.detect.normalize import normalize_token, token_forms
from video_beep_remover.models import Detection, Word


@dataclass(frozen=True)
class Token:
    raw: str
    normalized: str
    forms: tuple[str, ...]

    @classmethod
    def from_raw(cls, raw: str) -> "Token":
        normalized = normalize_token(raw)
        return cls(raw=raw, normalized=normalized, forms=token_forms(normalized))


@dataclass(frozen=True)
class Match:
    start: int  # index of the first matched token
    end: int  # exclusive
    term: str
    category: str
    targets: tuple[int, ...]  # token indexes to censor


def find_matches(lexicon: Lexicon, tokens: Sequence[Token]) -> list[Match]:
    """Longest match wins; ties go to the category listed first; each token is used at most once."""
    candidates: list[tuple[int, int, Match]] = []  # (length, term order, match)
    count = len(tokens)
    for start, token in enumerate(tokens):
        if not token.forms:
            continue
        for order, term in enumerate(lexicon.terms):
            length = len(term.words)
            if start + length > count:
                continue
            if not all(term.words[k].matches(tokens[start + k].forms) for k in range(length)):
                continue
            if length == 1 and lexicon.is_allowed(token.forms):
                continue
            targets = tuple(start + k for k in term.targets)
            candidates.append(
                (length, order, Match(start, start + length, term.text, term.category, targets))
            )

    candidates.sort(key=lambda c: (-c[0], c[1], c[2].start))
    taken = [False] * count
    selected: list[Match] = []
    for _, _, match in candidates:
        if any(taken[i] for i in range(match.start, match.end)):
            continue
        for i in range(match.start, match.end):
            taken[i] = True
        selected.append(match)

    for index, token in enumerate(tokens):
        if taken[index] or not lexicon.is_masked(token.normalized):
            continue
        category = lexicon.masked_category(token.normalized)
        if category is not None:
            label = token.normalized if category == MASKED_CATEGORY else f"masked:{token.normalized}"
            selected.append(Match(index, index + 1, label, category, (index,)))

    selected.sort(key=lambda m: m.start)
    return selected


def _runs(indexes: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(indexes):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def detect_in_words(lexicon: Lexicon, words: Sequence[Word]) -> list[Detection]:
    """One detection per contiguous run of censored words, timed by the words themselves."""
    tokens = [Token.from_raw(word.text) for word in words]
    detections: list[Detection] = []
    for match in find_matches(lexicon, tokens):
        for run in _runs(match.targets):
            span = [words[i] for i in run]
            detections.append(
                Detection(
                    start=span[0].start,
                    end=max(span[-1].end, span[0].start),
                    heard=" ".join(word.text.strip() for word in span),
                    term=match.term,
                    category=match.category,
                    confidence=min(word.probability for word in span),
                    source="asr",
                )
            )
    return detections
