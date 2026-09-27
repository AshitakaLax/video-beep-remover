"""Combine the signals into verdicts (DESIGN.md §17.4-17.5). Report-only: nothing here changes what
is muted.

The combination is one-sided, because letting a profane word through costs more than muting a
harmless one: a use is harmless only when the judge says so and the classifier finds the line clean;
a sexual line is never harmless; anything else undecided is "unsure", which mutes."""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

from video_beep_remover.context import rules
from video_beep_remover.context.lines import Line, heard_share, line_for, neighbours
from video_beep_remover.context.models import LABELS, Classifier, Judge
from video_beep_remover.models import Detection, Word

QUESTIONS_VERSION = 1  # bump when a question changes, so cached answers are not reused
Use = Literal["profane", "harmless", "unsure"]
_RUDE = ("toxicity", "obscene", "insult")
_REASONS = ("curse", "insult", "exclamation", "sexual", "literal", "religious", "place", "name", "other")
_HARMLESS_REASONS = ("literal", "religious", "place", "name", "other")
_EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
_JSON = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass(frozen=True)
class Settings:
    ambiguous: frozenset[str]  # terms, as written in the word list, whose sense is checked
    min_sexual_score: float  # classifier score from which a line counts as sexual
    clean_below: float  # a line is clean when every rude score is below this
    profane_above: float  # and clearly profane when one reaches this
    min_heard: float = 0.7  # the share of a subtitle line's words that must be heard to trust it


@dataclass(frozen=True)
class Verdict:
    use: Use
    reason: str
    line: int | None  # index into the lines
    scores: dict[str, float]
    emotion: str | None
    delivery: str | None
    intensity: str
    judged: bool

    @property
    def action(self) -> str:
        """What the layer would do; in report-only mode (M6), nothing is done."""
        return "keep" if self.use == "harmless" else "mute"


@dataclass(frozen=True)
class SexualLine:
    index: int
    line: Line
    evidence: tuple[str, ...]
    certain: bool  # False when the only evidence is weak: an ambiguous phrase or a sound
    score: float  # the classifier's sexual_explicit


@dataclass
class ContextResult:
    verdicts: list[Verdict]  # one per detection
    sexual: list[SexualLine]
    lines: list[Line]
    scores: list[dict[str, float]] = field(default_factory=list)


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _dialogue(lines: Sequence[Line], index: int, shown: Callable[[int], bool] = lambda i: True) -> str:
    before, after = neighbours(lines, index, shown)
    parts = [f"   {_quote(before)}"] if before else []
    parts.append(f">> {_quote(lines[index].text)}")
    if after:
        parts.append(f"   {_quote(after)}")
    return "\n".join(parts)


def sense_question(
    lines: Sequence[Line], index: int, word: str, shown: Callable[[int], bool] = lambda i: True
) -> str:
    return (
        f"Dialogue:\n{_dialogue(lines, index, shown)}\n\n"
        f"How is the word {_quote(word)} used in the line marked >>? Reply with "
        '{"use": ..., "reason": ..., "emotion": ...} where:\n'
        '- "use" is "profane" for a swear word, an insult, a curse, a sexual reference or an '
        'exclamation that takes God\'s name in vain, and "harmless" for a literal, reverent, place or '
        "name sense;\n"
        f'- "reason" is one of {", ".join(_quote(r) for r in _REASONS)};\n'
        f'- "emotion" is the speaker\'s emotion, one of {", ".join(_quote(e) for e in _EMOTIONS)}.'
    )


def sexual_question(lines: Sequence[Line], index: int) -> str:
    return (
        f"Dialogue:\n{_dialogue(lines, index)}\n\n"
        "Is the line marked >> sexual in nature, including innuendo? Reply with "
        '{"sexual": true} or {"sexual": false}.'
    )


def parse_answer(text: str) -> dict[str, Any]:
    """The first JSON object in a judge's answer, keeping only known fields with allowed values."""
    for match in _JSON.finditer(text):
        try:
            data = json.loads(match.group())
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        answer: dict[str, Any] = {}
        if data.get("use") in ("profane", "harmless"):
            answer["use"] = data["use"]
        if data.get("reason") in _REASONS:
            answer["reason"] = data["reason"]
        if data.get("emotion") in _EMOTIONS:
            answer["emotion"] = data["emotion"]
        if isinstance(data.get("sexual"), bool):
            answer["sexual"] = data["sexual"]
        return answer
    return {}


MASK = "[...]"


def mask_words(text: str, words: Sequence[str]) -> str:
    """The text with each of `words` (whole words, any case) replaced by MASK."""
    for word in sorted({w for w in words if w}, key=len, reverse=True):
        pattern = r"(?<![\w'])" + r"\W+".join(re.escape(part) for part in word.split()) + r"(?![\w'])"
        text = re.sub(pattern, MASK, text, flags=re.IGNORECASE)
    return text


def _score_lines(
    classifier: Classifier, lines: Sequence[Line], masked: dict[int, str]
) -> list[dict[str, float]]:
    """The classifier's scores for each line, or for its masked text where there is one: a classifier
    scores a word like "bitch" as rude in any sense, so an ambiguous listed word is scored without
    it, and the rest of the line decides (DESIGN.md §17.4)."""
    spoken = [i for i, line in enumerate(lines) if line.text]
    found = classifier.score([masked.get(i, lines[i].text) for i in spoken])
    scores = [dict.fromkeys(LABELS, 0.0) for _ in lines]
    for i, score in zip(spoken, found, strict=True):
        scores[i] = score
    return scores


def _masked_lines(
    detections: Sequence[Detection], lines: Sequence[Line], settings: Settings
) -> dict[int, str]:
    """For each line holding an ambiguous listed word, its text with those words masked."""
    words: dict[int, list[str]] = {}
    for detection in detections:
        index = line_for(detection, lines)
        if index is not None and detection.term.casefold() in settings.ambiguous:
            words.setdefault(index, []).append(detection.heard.strip(" ,.!?;:\"'"))
    masked = {index: mask_words(lines[index].text, found) for index, found in words.items()}
    return {index: text for index, text in masked.items() if text != lines[index].text}


def _sexual_lines(
    lines: Sequence[Line],
    scores: Sequence[dict[str, float]],
    judge: Judge | None,
    phrases: rules.Phrases,
    triggers: rules.Phrases,
    settings: Settings,
) -> list[SexualLine]:
    found = []
    for index, line in enumerate(lines):
        score = scores[index].get("sexual_explicit", 0.0)
        evidence: list[str] = []
        certain = False
        if score >= settings.min_sexual_score:
            evidence.append(f"classifier {score:.2f}")
            certain = True
        for term in phrases.find(line.text):
            weak = term.casefold() in settings.ambiguous
            evidence.append(f"phrase {_quote(term)}" + (" (ambiguous)" if weak else ""))
            certain = certain or not weak
        evidence += [f"sound [{sound}]" for sound in rules.sexual_sounds(line)]
        worth_asking = bool(triggers.find(line.text)) or (bool(evidence) and not certain)
        if (
            judge is not None
            and line.text
            and not certain
            and worth_asking
            and parse_answer(judge.ask(sexual_question(lines, index))).get("sexual") is True
        ):
            evidence.append("judge")
            certain = True
        if evidence:
            found.append(SexualLine(index, line, tuple(evidence), certain, score))
    return found


def _verdict(
    detection: Detection,
    lines: Sequence[Line],
    scores: Sequence[dict[str, float]],
    sexual: set[int],
    judge: Judge | None,
    settings: Settings,
    trusted: Callable[[int], bool],
) -> Verdict:
    index = line_for(detection, lines)
    line = lines[index] if index is not None else Line(detection.start, detection.end, detection.heard)
    score = scores[index] if index is not None else {}
    rude = max((score.get(label, 0.0) for label in _RUDE), default=0.0)

    def verdict(use: Use, reason: str, emotion: str | None = None, judged: bool = False) -> Verdict:
        return Verdict(
            use, reason, index, score, emotion, rules.delivery(line), rules.intensity(line), judged
        )

    if index is not None and index in sexual:
        return verdict("profane", "sexual line")
    if detection.term.casefold() not in settings.ambiguous:
        return verdict("profane", "listed")
    if index is None:
        return verdict("unsure", "no line to judge")
    word = detection.heard.strip(" ,.!?;:\"'") or detection.term
    if mask_words(line.text, [word]) == line.text:
        # Subtitles that soften what is said ("Go to heck" for "Go to hell") would be judged instead.
        return verdict("unsure", "the line does not show the word")
    if rude >= settings.profane_above:
        return verdict("profane", "classifier")
    if judge is None:
        return verdict("unsure", "not judged")  # the classifier alone never calls a use harmless
    if not trusted(index):
        # Text that was never said, such as a note written for the judge, must not decide (§17.9).
        return verdict("unsure", "the subtitles differ from what is heard")
    answer = parse_answer(judge.ask(sense_question(lines, index, word, trusted)))
    emotion = answer.get("emotion")
    if answer.get("use") == "harmless":
        if answer.get("reason") not in _HARMLESS_REASONS:
            return verdict("unsure", "the judge contradicted itself", emotion, True)
        if rude < settings.clean_below:
            return verdict("harmless", answer["reason"], emotion, True)
        return verdict("unsure", "the judge and the classifier disagree", emotion, True)
    if answer.get("use") == "profane":
        return verdict("profane", answer.get("reason", "judge"), emotion, True)
    return verdict("unsure", "no answer from the judge", emotion, True)


def analyse_context(
    detections: Sequence[Detection],
    lines: Sequence[Line],
    *,
    classifier: Classifier,
    judge: Judge | None,
    phrases: rules.Phrases,
    triggers: rules.Phrases,
    settings: Settings,
    heard: Sequence[Word] | None = None,
) -> ContextResult:
    """Verdicts for the detections, and the lines that look sexual. `heard` are the words heard in
    the audio: a subtitle line is trusted to show a use as harmless, or to be shown to the judge
    next to one, only if most of its words were heard (None: trust every line)."""

    @cache
    def trusted(index: int) -> bool:
        line = lines[index]
        return heard is None or line.cue is None or heard_share(line, heard) >= settings.min_heard

    scores = _score_lines(classifier, lines, _masked_lines(detections, lines, settings))
    sexual = _sexual_lines(lines, scores, judge, phrases, triggers, settings)
    certain = {s.index for s in sexual if s.certain}
    verdicts = [_verdict(d, lines, scores, certain, judge, settings, trusted) for d in detections]
    return ContextResult(verdicts, sexual, list(lines), scores)
