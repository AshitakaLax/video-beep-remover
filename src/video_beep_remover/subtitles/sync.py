"""Map subtitle time to media time and measure how verbatim the subtitles are (DESIGN.md §6.6)."""

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from rapidfuzz import fuzz

from video_beep_remover.config.schema import SyncConfig
from video_beep_remover.detect.normalize import normalize_token, split_words
from video_beep_remover.models import Cue, SyncModel, Word

MATCH_THRESHOLD = 75.0  # rapidfuzz ratio (0-100) above which an anchor cue counts as heard
MIN_SLOPE_SPAN_S = 60.0  # matched anchors must span this much before a scale is fitted, not just an offset
SNAP_TOLERANCE = 0.001  # fitted scales this close to a standard frame-rate ratio snap to it
MIN_SEARCH_S = 0.5
_FPS_PAIRS = ((25, 23.976), (25, 24), (24, 23.976), (30, 29.97))  # film, PAL and NTSC rates
STANDARD_RATIOS = (1.0, *(a / b for a, b in _FPS_PAIRS), *(b / a for a, b in _FPS_PAIRS))
# Anchor requirements: strict first, relaxed when a file has no cue that qualifies.
_ANCHOR_RULES = ((4, 1.0, 7.0), (2, 0.5, 10.0))  # (minimum words, shortest, longest in seconds)

WindowTranscriber = Callable[[float, float], Sequence[Word]]  # (start, end) in media time -> words


def cue_tokens(cue: Cue) -> list[str]:
    return [token for raw, _, _ in split_words(cue.text) if (token := normalize_token(raw))]


@dataclass(frozen=True)
class AnchorResult:
    cue: Cue
    start: float  # the searched stretch, in media time
    end: float
    score: float  # best rapidfuzz ratio, 0-100
    fidelity: float | None  # token_sort_ratio / 100 of that best match; None when nothing was heard
    heard_at: float | None  # media time of the first matched word, when the cue was heard

    @property
    def matched(self) -> bool:
        return self.heard_at is not None


@dataclass(frozen=True)
class SyncResult:
    model: SyncModel
    anchors: tuple[AnchorResult, ...]
    fidelity: float | None  # 0-1, the median over the anchors where speech was heard
    passed: bool
    reason: str | None = None  # why the check failed
    checked: bool = True  # False when `anchors = 0` turns the check off
    problem: Literal["no-anchors", "unmatched", "error", "fidelity"] | None = None  # the kind of failure

    @property
    def matched(self) -> int:
        return sum(anchor.matched for anchor in self.anchors)


def choose_anchors(cues: Sequence[Cue], count: int) -> list[Cue]:
    """Up to `count` cues spread over the runtime, one per equal stretch. Each has four or more words,
    lasts 1-7 s and is not sung. Within a stretch, the cue with the most rare words wins, because
    common lines ("Yes, sir.") could be matched in the wrong place."""
    if count <= 0 or not cues:
        return []
    tokens = {cue.index: cue_tokens(cue) for cue in cues}
    spread = Counter(token for words in tokens.values() for token in set(words))
    for min_words, shortest, longest in _ANCHOR_RULES:
        eligible = [
            cue
            for cue in cues
            if not cue.lyrics
            and shortest <= cue.end - cue.start <= longest
            and len(tokens[cue.index]) >= min_words
        ]
        if eligible:
            break
    else:
        return []

    def preference(cue: Cue) -> tuple[int, int, int]:
        words = tokens[cue.index]
        return (sum(1 for token in set(words) if spread[token] <= 2), len(words), -cue.index)

    first, last = cues[0].start, max(cue.end for cue in cues)
    stretch = (last - first) / count
    chosen = []
    for k in range(count):
        low, high = first + k * stretch, first + (k + 1) * stretch
        inside = [
            cue for cue in eligible if low <= cue.start < high or (k == count - 1 and cue.start >= high)
        ]
        if inside:
            chosen.append(max(inside, key=preference))
    return chosen


def best_match(expected: Sequence[str], words: Sequence[Word]) -> tuple[float, float | None, float | None]:
    """Find `expected` (normalized cue words) in recognized words by sliding a window of about the
    same length. Returns (ratio 0-100, fidelity 0-1, start time of the first matched word)."""
    heard = [(token, word.start) for word in words if (token := normalize_token(word.text))]
    if not expected or not heard:
        return 0.0, None, None
    target = " ".join(expected)
    sizes = sorted(
        {max(1, min(len(heard), size)) for size in (len(expected) - 1, len(expected), len(expected) + 1)}
    )
    best: tuple[float, float | None, float | None] = (-1.0, None, None)
    for size in sizes:
        for i in range(len(heard) - size + 1):
            text = " ".join(token for token, _ in heard[i : i + size])
            score = fuzz.ratio(target, text)
            if score > best[0]:
                best = (score, fuzz.token_sort_ratio(target, text) / 100, heard[i][1])
    return best


def snap(scale: float) -> float:
    for ratio in STANDARD_RATIOS:
        if abs(scale / ratio - 1) <= SNAP_TOLERANCE:
            return ratio
    return scale


def fit(pairs: Sequence[tuple[float, float]], *, default_scale: float = 1.0) -> SyncModel:
    """Fit media = scale * subtitle + offset to (subtitle time, media time) pairs.

    With three or more pairs that span at least MIN_SLOPE_SPAN_S, the scale is the Theil-Sen slope
    (the median of the pairwise slopes), snapped to a standard frame-rate ratio when close. Otherwise
    only the offset is fitted. The offset is the median residual and the error the median absolute
    residual, so a few mismatched anchors do not move the fit."""
    if not pairs:
        return SyncModel(scale=default_scale)
    scale = default_scale
    xs = [x for x, _ in pairs]
    if len(pairs) >= 3 and max(xs) - min(xs) >= MIN_SLOPE_SPAN_S:
        slopes = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in combinations(pairs, 2) if x2 != x1]
        if slopes:
            scale = snap(statistics.median(slopes))
    offset = statistics.median(y - scale * x for x, y in pairs)
    error = statistics.median(abs(y - (scale * x + offset)) for x, y in pairs)
    return SyncModel(scale=scale, offset=offset, error=error)


def check_sync(
    cues: Sequence[Cue],
    *,
    duration: float,
    transcribe: WindowTranscriber,
    config: SyncConfig,
    trusted: bool,
    default_scale: float = 1.0,
    on_anchor: Callable[[int], None] | None = None,
) -> SyncResult:
    """Transcribe a few seconds around each anchor cue and fit the timing (DESIGN.md §6.6).

    Anchors are searched in time order within ±`trusted_search_s` (or ±`untrusted_search_s`) of
    where the fit so far predicts them, so a drift that builds up over the film stays inside the
    search window. `on_anchor` receives the number of anchors done."""
    if config.anchors == 0:
        return SyncResult(SyncModel(scale=default_scale), (), None, passed=True, checked=False)
    anchors = choose_anchors(cues, config.anchors)
    if not anchors:
        return SyncResult(
            SyncModel(scale=default_scale),
            (),
            None,
            False,
            "no cue is usable as a sync anchor",
            problem="no-anchors",
        )

    radius = config.trusted_search_s if trusted else config.untrusted_search_s
    model = SyncModel(scale=default_scale)
    pairs: list[tuple[float, float]] = []
    results: list[AnchorResult] = []
    for done, cue in enumerate(anchors, 1):
        start = max(0.0, model.to_media(cue.start) - radius)
        end = min(duration, model.to_media(cue.end) + radius)
        if end - start < MIN_SEARCH_S:  # predicted outside the media
            results.append(AnchorResult(cue, start, max(start, end), 0.0, None, None))
        else:
            score, fidelity, heard_at = best_match(cue_tokens(cue), transcribe(start, end))
            if score >= MATCH_THRESHOLD and heard_at is not None:
                pairs.append((cue.start, heard_at))
                model = fit(pairs, default_scale=default_scale)
            else:
                heard_at = None
            results.append(AnchorResult(cue, start, end, score, fidelity, heard_at))
        if on_anchor is not None:
            on_anchor(done)

    heard = [anchor.fidelity for anchor in results if anchor.fidelity is not None]
    fidelity = statistics.median(heard) if heard else None
    matched = len(pairs)
    reason = None
    problem: Literal["unmatched", "error", "fidelity"] | None = None
    if matched / len(anchors) < config.min_matched_ratio:
        problem = "unmatched"
        reason = f"only {matched} of {len(anchors)} anchor cues were heard where the subtitles place them"
    elif model.error > config.max_error_s:
        problem = "error"
        reason = f"timing error {model.error:.2f} s is above max_error_s ({config.max_error_s} s)"
    elif fidelity is None or fidelity < config.min_fidelity:
        problem = "fidelity"
        shown = "unknown" if fidelity is None else f"{fidelity:.2f}"
        reason = (
            f"fidelity {shown} is below min_fidelity ({config.min_fidelity}): the subtitles are not verbatim"
        )
    return SyncResult(model, tuple(results), fidelity, passed=reason is None, reason=reason, problem=problem)
