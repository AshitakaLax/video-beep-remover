"""Many files in one run (DESIGN.md §8.2 item 7). The Whisper model stays loaded from file to file,
and each file is rendered in the background while the next one is analysed: rendering is mostly
FFmpeg reading and writing the whole file, analysis mostly speech recognition."""

import contextlib
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from video_beep_remover.errors import DependencyError, UsageError, VbrError
from video_beep_remover.media.probe import VIDEO_SUFFIXES
from video_beep_remover.pipeline import FileResult, Job, Pipeline, RunOptions, resolve_output
from video_beep_remover.ui import Progress


@dataclass(frozen=True)
class Input:
    path: Path
    from_folder: bool  # found by searching a folder, rather than named on the command line


def collect_inputs(paths: list[Path], recursive: bool) -> list[Input]:
    """The files named, and the videos in the folders named (sorted, each file once)."""
    found: list[Input] = []
    for path in paths:
        if path.is_dir():
            walk = path.rglob("*") if recursive else path.iterdir()
            found += [
                Input(p, True)
                for p in sorted(walk)
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and ".partial." not in p.name
            ]
        elif path.is_file():
            found.append(Input(path, False))
        else:
            raise UsageError(f"not found: {path}")
    unique: dict[Path, Input] = {}
    for item in found:
        resolved = item.path.resolve()
        if resolved not in unique or not item.from_folder:  # named on the command line wins
            unique[resolved] = Input(resolved, item.from_folder)
    if not unique:
        raise UsageError("no video files found")
    return list(unique.values())


def skip_outputs(
    inputs: list[Input], template: str, output: Path | None
) -> tuple[list[Input], list[FileResult]]:
    """Leave out files found in folders that are another input's output, e.g. movie.clean.mkv next to
    movie.mkv: cleaning it again would write movie.clean.clean.mkv. Files named on the command line
    are always processed."""
    outputs: dict[Path, Path] = {}
    for item in inputs:
        with contextlib.suppress(VbrError):
            outputs[resolve_output(item.path, template, output, many=True).resolve()] = item.path
    kept: list[Input] = []
    skipped: list[FileResult] = []
    for item in inputs:
        source = outputs.get(item.path)
        if item.from_folder and source is not None and source != item.path:
            skipped.append(FileResult(item.path, "skipped", notes=[f"it is the output for {source.name}"]))
        else:
            kept.append(item)
    return kept, skipped


class BufferedUI:
    """The UI of a render in the background: messages are kept for the main thread to show with the
    file's result, and nothing is drawn, since only one live display may run at a time."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []  # ("info" | "warn", message)

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warn(self, message: str) -> None:
        self.messages.append(("warn", message))

    @contextlib.contextmanager
    def progress(self, label: str, total: float) -> Iterator[Progress]:
        yield lambda done: None

    @contextlib.contextmanager
    def status(self, label: str) -> Iterator[None]:
        yield


@dataclass
class Outcome:
    path: Path
    result: FileResult | None = None
    error: VbrError | None = None
    messages: list[tuple[str, str]] = field(default_factory=list)  # from a render in the background


def _finish_in_background(pipeline: Pipeline, job: Job) -> Outcome:
    ui = BufferedUI()
    try:
        return Outcome(job.source, pipeline.finish(job, ui), messages=ui.messages)
    except DependencyError:
        raise  # stops the batch
    except VbrError as exc:
        return Outcome(job.source, error=exc, messages=ui.messages)


def run_batch(
    pipeline: Pipeline,
    inputs: list[Input],
    options: RunOptions,
    report: Callable[[Outcome], None],
    *,
    overlap: bool = True,
) -> None:
    """Process the inputs in order and pass each outcome to `report`, also in order. With `overlap`, a
    file is rendered in the background while the next one is analysed, one render at a time. A
    DependencyError (FFmpeg or a model missing) would fail every file, so it stops the batch."""
    many = len(inputs) > 1
    queue: deque[Outcome | Future[Outcome]] = deque()

    def flush(block: bool) -> None:
        while queue:
            slot = queue[0]
            if isinstance(slot, Future):
                if not (block or slot.done()):
                    return
                slot = slot.result()  # re-raises a DependencyError
            queue.popleft()
            report(slot)

    def wait_for_render() -> None:
        pending = [slot for slot in queue if isinstance(slot, Future) and not slot.done()]
        if pending:
            with pipeline.ui.status("Finishing the previous file's render"):
                wait(pending)
        flush(block=True)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vbr-render") as renders:
        for position, item in enumerate(inputs):
            try:
                prepared = pipeline.prepare(item.path, options, many=many, from_folder=item.from_folder)
            except DependencyError:
                raise
            except VbrError as exc:
                queue.append(Outcome(item.path, error=exc))
                flush(block=False)
                continue
            if isinstance(prepared, FileResult):
                queue.append(Outcome(item.path, prepared))
            else:
                try:
                    wait_for_render()  # one render at a time
                except BaseException:
                    pipeline.discard(prepared)
                    raise
                if prepared.render and overlap and position + 1 < len(inputs):
                    queue.append(renders.submit(_finish_in_background, pipeline, prepared))
                else:  # in the foreground, with a progress bar: nothing else is left to overlap
                    try:
                        queue.append(Outcome(item.path, pipeline.finish(prepared)))
                    except DependencyError:
                        raise
                    except VbrError as exc:
                        queue.append(Outcome(item.path, error=exc))
            flush(block=False)
        wait_for_render()
