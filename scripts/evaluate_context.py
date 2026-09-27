"""Score the context layer (DESIGN.md §17.7) on labelled lines.

    python scripts/evaluate_context.py [LINES.jsonl] [--judge MODEL | --no-judge] [--device cpu|cuda]
                                       [--json results.json]

Each line of the set is one JSON object, either a listed word's use or a line to check for sexual
content:

    {"text": "Go to hell!", "term": "hell", "use": "profane"}          (optionally "word" and "sounds")
    {"text": "Your place or mine?", "sexual": true}
    {"text": "", "sounds": ["moaning"], "sexual": true}

Each line is judged on its own, without neighbouring lines. The script reports:

    harmless precision  of the uses called harmless, the share that are; a profane use called harmless
                        is the costly error, since acting on the verdict would leave it audible
    harmless recall     of the harmless uses, the share called harmless
    unsure              uses left unsure, which mute like profane ones
    sexual recall       sexual lines flagged as certain, and flagged at all ("possible" included)
    false flags         other lines flagged as certain, and flagged at all
    judge               questions asked and seconds spent

The default set is scripts/data/context_lines.jsonl. Models come from the config, as in `vbr`; --judge
and --no-judge override context.judge."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from video_beep_remover.config import load_config
from video_beep_remover.context import ContextLayer
from video_beep_remover.context.lines import Line
from video_beep_remover.models import Detection

DEFAULT_SET = Path(__file__).parent / "data" / "context_lines.jsonl"


def read_set(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def ratio(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("lines", type=Path, nargs="?", default=DEFAULT_SET)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--judge", help="judge model (context.judge)")
    group.add_argument("--no-judge", action="store_true", help="rules and the classifier only")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    overrides: dict[str, Any] = {"transcription.device": args.device}
    if args.no_judge:
        overrides["context.judge"] = ""
    elif args.judge:
        overrides["context.judge"] = args.judge
    config = load_config(args.config, overrides=overrides).config
    device = args.device if args.device != "auto" else config.transcription.device
    if device == "auto":
        from video_beep_remover.asr.faster_whisper import cuda_available

        device = "cuda" if cuda_available() else "cpu"
    layer = ContextLayer(config, device=device, cache_dir=None)
    print(f"classifier {layer.classifier_name}, judge {layer.judge_name or 'none'}, on {device}")

    rows = read_set(args.lines)
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    questions = 0
    for row in rows:
        line = Line(0.0, 2.0, row.get("text", ""), tuple(row.get("sounds", [])), 1)
        if "use" in row:
            word = row.get("word", row["term"])
            detection = Detection(0.5, 0.8, f" {word}", row["term"], "listed", 0.9, "asr", 1)
            result, section = layer.run([detection], [line])
            verdict = result.verdicts[0]
            results.append(row | {"got": verdict.use, "reason": verdict.reason, "scores": verdict.scores})
        else:
            result, section = layer.run([], [line])
            flag = result.sexual[0] if result.sexual else None
            got = "certain" if flag and flag.certain else "possible" if flag else "no"
            results.append(row | {"got": got, "evidence": list(flag.evidence) if flag else []})
        questions += section["judge_questions"]
    seconds = time.monotonic() - started

    uses = [r for r in results if "use" in r]
    called = [r for r in uses if r["got"] == "harmless"]
    harmless = [r for r in uses if r["use"] == "harmless"]
    sexual = [r for r in results if r.get("sexual") is True]
    other = [r for r in results if r.get("sexual") is False]
    for r in results:
        truth = r.get("use") or ("sexual" if r.get("sexual") else "not sexual")
        why = r.get("reason") or "; ".join(r.get("evidence", []))
        print(f"{truth:>10} → {r['got']:<9} {r['text'] or r.get('sounds')}" + (f"  ({why})" if why else ""))
    summary = {
        "harmless precision": ratio(sum(r["use"] == "harmless" for r in called), len(called)),
        "harmless recall": ratio(len([r for r in harmless if r["got"] == "harmless"]), len(harmless)),
        "profane called harmless": sum(r["use"] == "profane" for r in called),
        "unsure": ratio(sum(r["got"] == "unsure" for r in uses), len(uses)),
        "sexual recall, certain": ratio(sum(r["got"] == "certain" for r in sexual), len(sexual)),
        "sexual recall, any": ratio(sum(r["got"] != "no" for r in sexual), len(sexual)),
        "false flags, certain": ratio(sum(r["got"] == "certain" for r in other), len(other)),
        "false flags, any": ratio(sum(r["got"] != "no" for r in other), len(other)),
        "judge questions": questions,
        "seconds": round(seconds, 1),
    }
    print()
    for name, value in summary.items():
        print(f"{name:>24}: {value}")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": results}, indent=2), "utf-8")


if __name__ == "__main__":
    main()
