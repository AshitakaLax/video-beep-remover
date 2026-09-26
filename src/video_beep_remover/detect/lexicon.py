"""Compile the configured word list into matchers (DESIGN.md §4.3)."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_beep_remover.config.schema import LexiconConfig
from video_beep_remover.detect.normalize import collapse_runs, fold
from video_beep_remover.errors import ConfigError

WILDCARD = r"[\w']*"
REGEX_PREFIX = "re:"
MASKED_CATEGORY = "masked"
# Endings tried when guessing which listed word a masked token such as "f***ing" stands for.
_SAMPLE_SUFFIXES = ("", "s", "es", "ed", "er", "ers", "ing", "in", "y", "ty", "head", "hole")
_MASK_CHUNK = re.compile(r"[\w']+|[^\w']+")


class TermError(ValueError):
    pass


@dataclass(frozen=True)
class WordPattern:
    text: str  # normalized pattern, e.g. "*fuck*"
    regex: re.Pattern[str]
    target: bool = False  # [bracketed] word: the part of a phrase that gets censored
    search: bool = False  # re: terms search inside the word; other patterns must match all of it

    def matches(self, forms: Sequence[str]) -> bool:
        test = self.regex.search if self.search else self.regex.fullmatch
        return any(test(form) is not None for form in forms)


@dataclass(frozen=True)
class Term:
    text: str  # as written in the config
    category: str
    words: tuple[WordPattern, ...]

    @property
    def targets(self) -> tuple[int, ...]:
        """Indexes of the words to censor: the bracketed ones, or all of them."""
        marked = tuple(i for i, word in enumerate(self.words) if word.target)
        return marked or tuple(range(len(self.words)))


def _normalize_pattern(word: str) -> str:
    return collapse_runs(fold(word).strip())


def parse_word(word: str) -> WordPattern:
    target = len(word) > 2 and word.startswith("[") and word.endswith("]")
    if target:
        word = word[1:-1]
    text = _normalize_pattern(word)
    if not text.replace("*", ""):
        raise TermError("a word needs at least one letter besides wildcards")
    variants = [text]
    if "-" in text and text.replace("-", "").replace("*", ""):
        variants.append(text.replace("-", ""))
    regex = "|".join(WILDCARD.join(re.escape(part) for part in variant.split("*")) for variant in variants)
    return WordPattern(text=text, regex=re.compile(regex), target=target)


def parse_term(text: str, category: str) -> Term:
    raw = text.strip()
    if not raw:
        raise TermError("empty term")
    if raw.startswith(REGEX_PREFIX):
        source = raw[len(REGEX_PREFIX) :]
        try:
            regex = re.compile(source, re.IGNORECASE)
        except re.error as exc:
            raise TermError(f"invalid regular expression: {exc}") from exc
        return Term(raw, category, (WordPattern(text=raw, regex=regex, search=True),))
    return Term(raw, category, tuple(parse_word(word) for word in raw.split()))


def _literal_letters(pattern: WordPattern) -> int:
    return sum(ch.isalpha() for ch in pattern.text)


def _samples(term: Term) -> tuple[str, ...]:
    words: list[str] = []
    for index in term.targets:
        pattern = term.words[index]
        if pattern.search:
            continue
        core = pattern.text.replace("*", "")
        words.extend(core + suffix for suffix in _SAMPLE_SUFFIXES)
    return tuple(words)


@dataclass(frozen=True)
class Lexicon:
    terms: tuple[Term, ...]  # enabled categories, in config order
    allow: tuple[WordPattern, ...]
    hints: tuple[Term, ...]
    masked_patterns: tuple[re.Pattern[str], ...]  # empty when masked detection is off
    category_samples: tuple[tuple[str, bool, tuple[str, ...]], ...]  # (category, enabled, samples)
    categories: tuple[str, ...]  # enabled category names, in config order
    warnings: tuple[str, ...]

    def is_allowed(self, forms: Sequence[str]) -> bool:
        return any(pattern.matches(forms) for pattern in self.allow)

    def is_masked(self, normalized: str) -> bool:
        if not self.masked_patterns or len(normalized) < 2:
            return False
        if not any(pattern.search(normalized) for pattern in self.masked_patterns):
            return False
        return any(ch.isalpha() for ch in normalized) or len(normalized) >= 3

    def masked_category(self, normalized: str) -> str | None:
        """Category of the listed word a masked token stands for.

        Returns MASKED_CATEGORY when no visible letters identify it, and None when it matches
        only a disabled category (so a word the user chose not to censor stays audible).
        """
        if not any(ch.isalpha() for ch in normalized):
            return MASKED_CATEGORY
        chunks = _MASK_CHUNK.findall(normalized)

        def build(mask: str) -> re.Pattern[str]:
            return re.compile(
                "".join(
                    re.escape(chunk) if chunk[0].isalnum() or chunk[0] in "_'" else mask.format(n=len(chunk))
                    for chunk in chunks
                )
            )

        # Masks usually hide one letter per character ("d**n" is "damn"); try that before any length.
        for pattern in (build(r"[\w']{{{n}}}"), build(r"[\w']+")):
            for category, enabled, samples in self.category_samples:
                if any(pattern.fullmatch(sample) for sample in samples):
                    return category if enabled else None
        return MASKED_CATEGORY


def read_word_file(path: Path) -> list[str]:
    terms = []
    for line in path.read_text("utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        terms.append(re.split(r"\s#", text, maxsplit=1)[0].strip())
    return terms


def compile_lexicon(
    config: LexiconConfig,
    *,
    only: Iterable[str] | None = None,
    base_dir: Path | None = None,
) -> Lexicon:
    """Compile enabled categories. `only` (from --categories) enables exactly the named categories."""
    wanted = None if only is None else [name.strip() for name in only if name.strip()]
    sources: list[tuple[str, bool, list[str]]] = [
        (name, category.enabled, list(category.terms)) for name, category in config.categories.items()
    ]
    for file in config.files:
        path = Path(file).expanduser()
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        if not path.is_file():
            raise ConfigError(f"lexicon.files: word file not found: {path}")
        sources.append((path.stem, True, read_word_file(path)))

    known = [name for name, _, _ in sources]
    if wanted is not None:
        unknown = sorted(set(wanted) - set(known))
        if unknown:
            raise ConfigError(
                f"unknown categories: {', '.join(unknown)} (available: {', '.join(known) or 'none'})"
            )
        sources = [(name, name in wanted, terms) for name, _, terms in sources]

    terms: list[Term] = []
    samples: list[tuple[str, bool, tuple[str, ...]]] = []
    warnings: list[str] = []
    for name, enabled, raw_terms in sources:
        compiled: list[Term] = []
        for raw in raw_terms:
            try:
                term = parse_term(raw, name)
            except TermError as exc:
                raise ConfigError(f"lexicon category {name!r}, term {raw!r}: {exc}") from exc
            compiled.append(term)
            for word in term.words:
                if not word.search and "*" in word.text and _literal_letters(word) < 3:
                    warnings.append(
                        f"term {raw!r} in category {name!r} has fewer than three literal letters "
                        "and may match many innocent words"
                    )
        samples.append((name, enabled, tuple(s for term in compiled for s in _samples(term))))
        if enabled:
            terms.extend(compiled)

    allow: list[WordPattern] = []
    for raw in config.allow:
        if len(raw.split()) != 1:
            raise ConfigError(f"lexicon.allow: {raw!r}: allowlist entries are single words")
        try:
            allow.append(parse_word(raw))
        except TermError as exc:
            raise ConfigError(f"lexicon.allow: {raw!r}: {exc}") from exc

    hints: list[Term] = []
    for raw in config.hints.terms:
        try:
            hints.append(parse_term(raw, "hint"))
        except TermError as exc:
            raise ConfigError(f"lexicon.hints: {raw!r}: {exc}") from exc

    masked: list[re.Pattern[str]] = []
    if config.detect_masked:
        for raw in config.masked_patterns:
            try:
                masked.append(re.compile(raw))
            except re.error as exc:
                raise ConfigError(f"lexicon.masked_patterns: {raw!r}: {exc}") from exc

    return Lexicon(
        terms=tuple(terms),
        allow=tuple(allow),
        hints=tuple(hints),
        masked_patterns=tuple(masked),
        category_samples=tuple(samples),
        categories=tuple(name for name, enabled, _ in sources if enabled),
        warnings=tuple(dict.fromkeys(warnings)),
    )
