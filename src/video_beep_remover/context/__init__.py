"""Context analysis (DESIGN.md §17): verdicts on the detections, and the lines that look sexual, for
the report and the review subtitles. Acting on them is opt-in (M7): context.harmless = "keep" leaves
uses judged harmless unmuted, and context.sexual = "mute" mutes the lines flagged as sexual."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from video_beep_remover.config.schema import Config, ContextApiConfig
from video_beep_remover.context import rules
from video_beep_remover.context.analyse import (
    QUESTIONS_VERSION,
    Choice,
    ContextResult,
    Settings,
    SexualLine,
    Verdict,
    analyse_context,
    choose_substitutes,
)
from video_beep_remover.context.lines import Line, build_lines, word_lines
from video_beep_remover.context.models import (
    DEFAULT_JUDGE,
    JUDGE_GPU_GB,
    CachedJudge,
    Classifier,
    Judge,
    LocalJudge,
    ToxicityClassifier,
    check_installed,
    gpu_memory_gb,
    torch_device,
)
from video_beep_remover.models import CensorInterval, Detection, Word

__all__ = [
    "Choice",
    "ContextLayer",
    "ContextResult",
    "Line",
    "ModelFactory",
    "build_lines",
    "check_installed",
    "gpu_memory_gb",
    "judge_model",
    "judge_off_reason",
    "kept_cues",
    "review_cues",
    "review_labels",
    "review_notes",
    "torch_device",
    "verdict_dict",
    "word_lines",
]

ModelFactory = Callable[[str, str], Any]  # (model name, device) -> a Classifier or a Judge


def judge_model(
    setting: str,
    device: str,
    api: ContextApiConfig | None = None,
    *,
    offline: bool = False,
    gpu_gb: float | None = None,
) -> str | None:
    """The judge to run: "auto" runs the default one only on a GPU, where it is fast, and only if the GPU
    has room for it (`gpu_gb`, its memory, when known); "api" the service of context.api (DESIGN.md
    §17.10), except offline."""
    if not setting:
        return None
    if setting == "auto":
        fits = gpu_gb is None or gpu_gb >= JUDGE_GPU_GB
        return DEFAULT_JUDGE if device == "cuda" and fits else None
    if setting == "api":
        from video_beep_remover.context.api import judge_name

        return None if offline else judge_name(api or ContextApiConfig())
    return setting


def judge_off_reason(setting: str, device: str, gpu_gb: float | None) -> str:
    """Why judge_model chose no judge, for the report."""
    if not setting:
        return 'turned off (context.judge = "")'
    if setting == "api":
        return 'offline: context.judge = "api" needs the network'
    if device == "cuda" and gpu_gb is not None:
        return (
            f'the GPU has {gpu_gb:.0f} GB: context.judge = "auto" needs about {JUDGE_GPU_GB:.0f} GB for '
            f'{DEFAULT_JUDGE}; "api" asks a service instead'
        )
    return 'no GPU: context.judge = "auto" runs the judge only on an NVIDIA GPU'


class ContextLayer:
    """The context layer for a run: its models load on first use and serve every file of a batch."""

    def __init__(
        self,
        config: Config,
        *,
        device: str,
        cache_dir: Path | None,
        classifier_factory: ModelFactory | None = None,
        judge_factory: ModelFactory | None = None,
    ) -> None:
        settings = config.context
        self.settings = Settings(
            ambiguous=frozenset(term.casefold() for term in settings.ambiguous),
            min_sexual_score=settings.min_sexual_score,
            clean_below=settings.clean_below,
            profane_above=settings.profane_above,
            min_heard=settings.min_heard,
        )
        sexual = config.lexicon.categories.get("sexual")
        self.phrases = rules.Phrases(sexual.terms if sexual else [], "sexual")
        self.triggers = rules.Phrases(settings.triggers, "trigger")
        self.device = device
        self.harmless = settings.harmless
        self.sexual = settings.sexual
        self.classifier_name = settings.classifier
        self.judge_setting = settings.judge
        gpu_gb = gpu_memory_gb() if settings.judge == "auto" and device == "cuda" else None
        self.judge_name = judge_model(
            settings.judge, device, settings.api, offline=config.offline, gpu_gb=gpu_gb
        )
        self.judge_off = None if self.judge_name else judge_off_reason(settings.judge, device, gpu_gb)
        self.cache_dir = cache_dir
        offline = config.offline
        self._classifier_factory = classifier_factory or (
            lambda name, where: ToxicityClassifier(name, device=where, offline=offline)
        )
        self._judge_factory = judge_factory or (
            (lambda name, where: _api_judge(settings.api))
            if settings.judge == "api"
            else (lambda name, where: LocalJudge(name, device=where, offline=offline))
        )
        self._classifier: Classifier | None = None
        self._judge: CachedJudge | None = None

    def classifier(self) -> Classifier:
        if self._classifier is None:
            self._classifier = self._classifier_factory(self.classifier_name, self.device)
        return self._classifier

    def judge(self) -> CachedJudge | None:
        if self.judge_name is None:
            return None
        if self._judge is None:
            model: Judge = self._judge_factory(self.judge_name, self.device)
            self._judge = CachedJudge(model, self.cache_dir, QUESTIONS_VERSION)
        return self._judge

    def release(self) -> bool:
        """Drop the classifier and a local judge, so their memory can be freed; their next use loads them
        again. A judge behind an API holds no model, so it stays, with its connection and its answers.
        Returns whether anything was dropped."""
        keep_judge = self.judge_setting == "api"
        dropped = self._classifier is not None or (self._judge is not None and not keep_judge)
        self._classifier = None
        if not keep_judge:
            self._judge = None
        return dropped

    def run(
        self, detections: Sequence[Detection], lines: Sequence[Line], heard: Sequence[Word] | None = None
    ) -> tuple[ContextResult, dict[str, Any]]:
        """Verdicts and sexual lines, and the report's `context` section. `heard`: the words heard in
        the audio, against which subtitle lines are checked (None: trust every line)."""
        judge = self.judge()
        asked, seconds, unanswered = (judge.asked, judge.seconds, judge.unanswered) if judge else (0, 0.0, 0)
        # A judge behind an API counts the answers of each of its models (context.api.fallback_models).
        answered: Counter[str] | None = getattr(judge.judge, "answered", None) if judge else None
        before = Counter(answered or {})
        result = analyse_context(
            detections,
            lines,
            classifier=self.classifier(),
            judge=judge,
            phrases=self.phrases,
            triggers=self.triggers,
            settings=self.settings,
            heard=heard,
        )
        uses = [v.use for v in result.verdicts]
        section = {
            "classifier": self.classifier_name,
            "judge": self.judge_name,
            "harmless": self.harmless,
            "sexual": self.sexual,
            "lines": len(lines),
            "verdicts": {use: uses.count(use) for use in ("profane", "harmless", "unsure")},
            "judge_questions": (judge.asked - asked) if judge else 0,
            "judge_unanswered": (judge.unanswered - unanswered) if judge else 0,
            "judge_seconds": round(judge.seconds - seconds, 1) if judge else 0.0,
            "sexual_lines": [_sexual_dict(s) for s in result.sexual],
        }
        if answered is not None:
            section["judge_models"] = dict(answered - before)
        if self.judge_off is not None:
            section["judge_off"] = self.judge_off
        return result, section

    def substitutes(
        self, detections: Sequence[Detection], result: ContextResult, table: Mapping[str, Sequence[str]]
    ) -> list[Choice]:
        """The substitute to say in place of each detection, if any (DESIGN.md §17.6)."""
        return choose_substitutes(detections, result, table, self.judge())


def _api_judge(api: ContextApiConfig) -> Any:
    from video_beep_remover.context.api import api_judge

    return api_judge(api)


def verdict_dict(verdict: Verdict, lines: Sequence[Line]) -> dict[str, Any]:
    line = lines[verdict.line] if verdict.line is not None else None
    return {
        "use": verdict.use,
        "reason": verdict.reason,
        "action": verdict.action,
        "line": line.text if line else None,
        "scores": {label: round(value, 3) for label, value in verdict.scores.items()},
        "emotion": verdict.emotion,
        "delivery": verdict.delivery,
        "intensity": verdict.intensity,
        "judged": verdict.judged,
    }


def _sexual_dict(found: SexualLine) -> dict[str, Any]:
    return {
        "start": round(found.line.start, 3),
        "end": round(found.line.end, 3),
        "text": found.line.text,
        "sounds": list(found.line.sounds),
        "cue": found.line.cue,
        "evidence": list(found.evidence),
        "certain": found.certain,
        "score": round(found.score, 3),
    }


def review_notes(verdicts: Sequence[dict[str, Any] | None]) -> list[str | None]:
    """A note per detection for the review subtitles, from the report's verdicts: nothing for a plain
    profane use."""
    notes: list[str | None] = []
    for verdict in verdicts:
        use, reason = (verdict or {}).get("use"), (verdict or {}).get("reason")
        if use == "harmless":
            notes.append(f"probably harmless: {reason}")
        elif use == "unsure":
            notes.append(f"unsure: {reason}")
        elif reason == "sexual line":
            notes.append("sexual line")
        else:
            notes.append(None)
    return notes


def _evidence(item: dict[str, Any]) -> str:
    return "; ".join(str(e) for e in item.get("evidence") or [])


def _span(item: Any) -> tuple[float, float] | None:
    try:
        return float(item["start"]), float(item["end"])
    except (TypeError, KeyError, ValueError):
        return None


def _covered(span: tuple[float, float] | None, intervals: Sequence[CensorInterval]) -> bool:
    return span is not None and any(i.start < span[1] and span[0] < i.end for i in intervals)


def review_cues(
    section: dict[str, Any] | None, intervals: Sequence[CensorInterval] = ()
) -> list[tuple[float, float, str]]:
    """A review cue for each line the report flags as sexual, unless it was muted (and still is: a
    span can be deleted from the report by hand)."""
    cues = []
    for item in (section or {}).get("sexual_lines") or []:
        span = _span(item)
        if span is None or _covered(_span(item.get("muted")), intervals):
            continue
        label = "[sexual line]" if item.get("certain") else "[possibly sexual]"
        cues.append((*span, f"{label} {_evidence(item)}".strip()))
    return cues


def review_labels(section: dict[str, Any] | None) -> list[tuple[float, float, str]]:
    """Labels for the spans muted because a line was flagged as sexual (context.sexual = "mute")."""
    labels = []
    for item in (section or {}).get("sexual_lines") or []:
        span = _span(item.get("muted"))
        if span is not None:
            labels.append((*span, f"sexual line ({_evidence(item)})"))
    return labels


def kept_cues(
    detections: Sequence[Detection],
    verdicts: Sequence[dict[str, Any] | None],
    intervals: Sequence[CensorInterval],
) -> list[tuple[float, float, str]]:
    """A review cue for each use judged harmless that no muted span covers: context.harmless = "keep"
    left it audible."""
    cues = []
    for detection, verdict in zip(detections, verdicts, strict=False):
        if (verdict or {}).get("use") != "harmless":
            continue
        if any(i.start < detection.end and detection.start < i.end for i in intervals):
            continue
        text = f"[kept] {detection.heard.strip()} (probably harmless: {(verdict or {}).get('reason')})"
        cues.append((detection.start, detection.end, text))
    return cues
