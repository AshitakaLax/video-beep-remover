"""Score vbr against clips whose listed words are annotated (DESIGN.md §11).

    python scripts/evaluate.py SET_DIR [--strategy full --strategy hybrid ...] [--set KEY=VALUE ...]
                               [--config vbr.toml] [--json results.json]

SET_DIR holds videos, each with a ground-truth file next to it:

    Movie.mkv  Movie.truth.json  {"words": [{"start": 12.31, "end": 12.62, "word": "hell"}, ...]}
    Movie.en.srt                 subtitles, if the clip has any (found like any sidecar)

Every strategy runs on every clip (a dry run, with the transcript cache off so each configuration is
timed on its own). The table shows, per strategy:

    recall      listed words muted over at least 95 % of their length (the primary metric)
    partial     listed words muted over at least half their length
    precision   detections that overlap a listed word
    start/end   median and worst distance between a detected word's edges and the annotation
    extra       seconds muted that are not a listed word, per minute of video
    audio       seconds of audio transcribed, per minute of video
    time        wall time, per minute of video

--set KEY=VALUE overrides a config key for the run, e.g. --set censor.pad_before_ms=80. --cache DIR keeps
transcripts in DIR, so runs that differ only in settings the transcripts do not depend on (padding,
word list, windows) need no speech recognition after the first; their times are then not comparable."""

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_beep_remover.config import load_config
from video_beep_remover.media.probe import VIDEO_SUFFIXES
from video_beep_remover.pipeline import Pipeline, RunOptions

FULL = 0.95
PARTIAL = 0.5


@dataclass
class Score:
    words: int = 0
    muted: int = 0
    partly: int = 0
    detections: int = 0
    true_detections: int = 0
    start_errors: list[float] = field(default_factory=list)
    end_errors: list[float] = field(default_factory=list)
    extra_seconds: float = 0.0
    audio_seconds: float = 0.0
    duration: float = 0.0
    wall: float = 0.0
    missed: list[str] = field(default_factory=list)
    false: list[str] = field(default_factory=list)

    def add(self, other: "Score") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score_clip(name: str, truth: list[dict[str, Any]], report: dict[str, Any], wall: float) -> Score:
    words = [(float(w["start"]), float(w["end"]), str(w["word"])) for w in truth]
    intervals = [(float(i["start"]), float(i["end"])) for i in report["intervals"]]
    detections = report["detections"]
    duration = float(report["input"]["duration"])
    score = Score(words=len(words), detections=len(detections), duration=duration, wall=wall)
    for start, end, word in words:
        covered = sum(overlap((start, end), span) for span in intervals) / max(1e-9, end - start)
        score.muted += covered >= FULL
        score.partly += covered >= PARTIAL
        if covered < FULL:
            score.missed.append(f"{name} {start:.2f} {word} ({covered:.0%} muted)")
        found = [d for d in detections if overlap((start, end), (d["start"], d["end"])) > 0]
        if found:
            best = min(found, key=lambda d: abs(d["start"] - start))
            score.start_errors.append(best["start"] - start)
            score.end_errors.append(best["end"] - end)
    for d in detections:
        if any(overlap((d["start"], d["end"]), (s, e)) > 0 for s, e, _ in words):
            score.true_detections += 1
        else:
            score.false.append(f"{name} {d['start']:.2f} {d['heard'].strip()} ({d['source']})")
    muted = sum(e - s for s, e in intervals)
    on_words = sum(overlap((s, e), span) for s, e, _ in words for span in intervals)
    score.extra_seconds = muted - on_words
    windows = report.get("windows") or {}
    score.audio_seconds = float(windows.get("audio_seconds", duration)) if windows else duration
    if report["strategy"]["used"] == "full":
        score.audio_seconds = duration
    return score


def run(
    clips: list[Path], strategy: str, overrides: dict[str, Any], config: Path | None, cache: Path | None
) -> tuple[Score, list[dict[str, Any]]]:
    caching = {"cache.transcripts": True, "cache.dir": str(cache)} if cache else {"cache.transcripts": False}
    settings = {"analysis.strategy": strategy, **caching, **overrides}
    pipeline = Pipeline(load_config(config, overrides=settings))
    total = Score()
    rows = []
    with tempfile.TemporaryDirectory(prefix="vbr-eval-") as tmp:
        for clip in clips:
            truth = json.loads(clip.with_name(f"{clip.stem}.truth.json").read_text("utf-8"))["words"]
            report_path = Path(tmp) / f"{clip.stem}.json"
            started = time.monotonic()
            pipeline.process(clip, RunOptions(dry_run=True, report=report_path))
            wall = time.monotonic() - started
            report = json.loads(report_path.read_text("utf-8"))
            score = score_clip(clip.stem, truth, report, wall)
            total.add(score)
            rows.append(
                {
                    "clip": clip.stem,
                    "used": report["strategy"]["used"],
                    "fallback_reason": report["strategy"]["fallback_reason"],
                    "words": score.words,
                    "muted": score.muted,
                    "detections": score.detections,
                    "false": score.false,
                    "missed": score.missed,
                    "wall": round(wall, 1),
                }
            )
    return total, rows


def ms(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values) * 1000:+.0f}/{max(values, key=abs) * 1000:+.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", type=Path)
    parser.add_argument("--strategy", action="append", choices=["full", "hybrid", "targeted"])
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--cache", type=Path, help="keep transcripts here between runs")
    args = parser.parse_args()
    overrides: dict[str, Any] = {}
    for item in args.overrides:
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)  # numbers, true/false, lists
        except ValueError:
            overrides[key] = value
    clips = sorted(
        path
        for path in args.set.iterdir()
        if path.suffix.lower() in VIDEO_SUFFIXES and path.with_name(f"{path.stem}.truth.json").is_file()
    )
    if not clips:
        raise SystemExit(f"no annotated clips in {args.set}")
    columns = [("strategy", 10), ("recall", 8), ("partial", 9), ("precision", 11), ("start ms", 11),
               ("end ms", 10), ("extra", 7), ("audio", 7), ("time", 7)]  # fmt: skip
    lines = ["".join(f"{name:<{w}}" if i == 0 else f"{name:>{w}}" for i, (name, w) in enumerate(columns))]
    results = {}
    total = Score()
    for strategy in args.strategy or ["full", "targeted", "hybrid"]:
        total, rows = run(clips, strategy, overrides, args.config, args.cache)
        minutes = total.duration / 60
        cells = [
            strategy,
            f"{total.muted / total.words:.1%}",
            f"{total.partly / total.words:.1%}",
            f"{total.true_detections / max(1, total.detections):.1%}",
            ms(total.start_errors),
            ms(total.end_errors),
            f"{total.extra_seconds / minutes:.1f}",
            f"{total.audio_seconds / minutes:.1f}",
            f"{total.wall / minutes:.1f}",
        ]
        lines.append(
            "".join(
                f"{c:<{w}}" if i == 0 else f"{c:>{w}}"
                for i, (c, (_, w)) in enumerate(zip(cells, columns, strict=True))
            )
        )
        results[strategy] = {"total": vars(total), "clips": rows}
        for row in rows:
            if row["used"] != strategy:
                print(f"  {strategy}: {row['clip']} used {row['used']}: {row['fallback_reason']}")
        for item in total.missed:
            print(f"  {strategy}: not fully muted: {item}")
        for item in total.false:
            print(f"  {strategy}: not a listed word: {item}")
    print(f"\n{len(clips)} clips, {total.words} listed words, {total.duration / 60:.1f} min")
    print(f"overrides: {overrides or 'none'}")
    print("start/end ms: median/worst of detected minus annotated")
    print("extra, audio, time: seconds per minute of video")
    print("\n".join(lines))
    if args.json:
        args.json.write_text(
            json.dumps({"overrides": overrides, "results": results}, indent=1) + "\n", "utf-8"
        )


if __name__ == "__main__":
    main()
