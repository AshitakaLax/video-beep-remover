"""Voice replacement in the pipeline (DESIGN.md §16), with real FFmpeg and stand-in models: a pure tone
stands in for speech, and the stand-in voice model says a word again as a 1 kHz tone, which the
stand-in speech recognition hears as the substitute."""

import json
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers import MUSIC_HZ, FakeTranscriber, StrictUI, Track, decode, make_clip, tone_gain, words
from video_beep_remover.asr.base import Clip
from video_beep_remover.config import load_config
from video_beep_remover.context.models import LABELS
from video_beep_remover.media.audio import read_pcm_windows
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe, select_audio_stream
from video_beep_remover.models import Word
from video_beep_remover.pipeline import Pipeline, RunOptions

pytestmark = pytest.mark.ffmpeg

SPOKEN = words(("this", 1.0, 1.2), ("is", 1.25, 1.4), ("damn", 2.0, 2.4), ("good.", 2.5, 2.8))
NEW_WORD_HZ = 1000


class Classifier:
    name = "fake-classifier"

    def score(self, texts: Sequence[str]) -> list[dict[str, float]]:
        return [dict.fromkeys(LABELS, 0.0) for _ in texts]


class Hearing(FakeTranscriber):
    """Hears the scripted words, except that a word said again as the 1 kHz tone is heard as "darn"."""

    def transcribe(self, clips: Sequence[Clip], **options: Any) -> list[list[Word]]:
        heard = super().transcribe(clips, **options)
        found = []
        for clip, clip_words in zip(clips, heard, strict=True):
            rate = 16_000
            said = []
            for word in clip_words:
                block = clip.audio[
                    int((word.start - clip.start) * rate) : int((word.end - clip.start) * rate)
                ]
                n = np.arange(block.size)
                level = (
                    np.abs(np.dot(block, np.exp(-2j * np.pi * NEW_WORD_HZ * n / rate)))
                    * 2
                    / max(1, block.size)
                )
                said.append(Word(" darn", word.start, word.end, 0.9) if level > 0.05 else word)
            found.append(said)
        return found


class Separator:
    name = "fake-separator"

    def vocals(self, audio: Any, rate: int) -> Any:
        return audio  # the tone is all dialogue


class Editor:
    name = "fake-editor"

    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.texts: list[str] = []

    def edit(self, voice: Any, rate: int, text: str, span: tuple[float, float]) -> Any:
        self.texts.append(text)
        edited = voice.copy()
        if self.works:
            first, last = round(span[0] * rate), round(span[1] * rate)
            n = np.arange(last - first)
            edited[first:last] = 0.125 * np.sin(2 * np.pi * NEW_WORD_HZ * n / rate)
        return edited


class Encoder:
    def embed(self, voice: Any, rate: int) -> Any:
        return np.ones(4, dtype=np.float32)


def run(
    tmp_path: Path,
    editor: Editor,
    tracks: Sequence[Track] = (Track(default=True), Track(title="Downmix")),  # the same dialogue twice
    audio_codec: str = "flac",
    name: str = "movie.mkv",
    **overrides: Any,
) -> tuple[Pipeline, Any]:
    loaded = load_config(
        None,
        env={},
        cwd=tmp_path,
        overrides={"transcription.device": "cpu", "replace.enabled": True, "context.judge": "", **overrides},
    )
    fake = Hearing(SPOKEN)
    pipeline = Pipeline(
        loaded,
        ui=StrictUI(),
        transcriber_factory=lambda choice: fake,
        context_models=(lambda name, device: Classifier(), lambda name, device: None),
        voice_models=(Separator, lambda: editor, Encoder),
    )
    source = make_clip(tmp_path / name, audio_codec=audio_codec, tracks=tracks)
    return pipeline, pipeline.process(source, RunOptions(review_srt=True, edl=True))


def test_a_replaced_word_is_said_again_instead_of_muted(tmp_path: Path) -> None:
    editor = Editor()
    pipeline, result = run(tmp_path, editor)
    assert (result.status, result.replaced) == ("cleaned", 1)
    assert editor.texts == ["this is darn good."]

    samples = decode(result.output)
    # 100 ms windows: shorter ones cannot tell 440 Hz from 1 kHz.
    assert tone_gain(samples, 2.2, window=0.1) < 0.01  # the old word is gone...
    assert tone_gain(samples, 2.2, frequency=NEW_WORD_HZ, window=0.1) == pytest.approx(
        1.0, abs=0.02
    )  # ...for the new
    assert tone_gain(samples, 1.1) == pytest.approx(1.0, abs=0.1)  # the rest is untouched
    assert tone_gain(samples, 2.9) == pytest.approx(1.0, abs=0.1)
    downmix = decode(result.output, "0:a:1")  # the new word was made for the analysed stream only
    assert tone_gain(downmix, 2.2) < 0.01 and tone_gain(downmix, 2.2, frequency=NEW_WORD_HZ) < 0.01

    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    [replacement] = report["replacements"]
    assert (replacement["word"], replacement["substitute"], replacement["replaced"]) == ("damn", "darn", True)
    assert replacement["heard"] == "darn"
    assert report["detections"][0]["context"]["substitute"] == "darn"
    assert report["intervals"] == [{"start": 1.88, "end": 2.6}]  # still muted by the EDL and vbr render
    assert report["output"]["verified_spans"] == 1  # in the downmix: the analysed stream holds the new word
    assert "[replaced] damn → darn" in (tmp_path / "movie.clean.review.srt").read_text("utf-8")
    assert (tmp_path / "movie.edl").read_text("utf-8") == "1.880\t2.600\t1\n"

    # vbr render cannot replace words: from the report, it mutes the span.
    rendered = pipeline.render_report(
        tmp_path / "movie.mkv", tmp_path / "movie.clean.vbr.json", RunOptions(output=tmp_path / "again.mkv")
    )
    again = decode(rendered.output)  # type: ignore[arg-type]
    assert tone_gain(again, 2.2) < 0.01 and tone_gain(again, 2.2, frequency=NEW_WORD_HZ) < 0.01


def test_a_surround_track_is_edited_in_its_front_centre(tmp_path: Path) -> None:
    # The dialogue tone is in the front centre, and the other channels carry "music". The check hears
    # the track through FFmpeg's downmix, as the analysis does; an even mix of the six channels would
    # bury the new word. AC3 starts 256 samples early, before zero: the old tone cancels only if the
    # change is added at the exact sample.
    _, result = run(tmp_path, Editor(), tracks=(Track(default=True, surround=True),), audio_codec="ac3")
    assert (result.status, result.replaced) == ("cleaned", 1)
    centre, left = decode(result.output, channel="FC"), decode(result.output, channel="FL")
    assert tone_gain(centre, 2.2, window=0.1) < 0.01
    assert tone_gain(centre, 2.2, frequency=NEW_WORD_HZ, window=0.1) == pytest.approx(1.0, abs=0.02)
    assert tone_gain(centre, 1.1, window=0.1) == pytest.approx(1.0, abs=0.02)
    # The music under the word stays.
    assert tone_gain(left, 2.2, frequency=MUSIC_HZ, window=0.1) == pytest.approx(1.0, abs=0.02)
    assert tone_gain(left, 2.2, frequency=NEW_WORD_HZ, window=0.1) < 0.01


def test_a_word_is_said_again_in_an_mp4(tmp_path: Path) -> None:
    # AAC in MP4, as most downloaded films are: the old tone cancels only at the exact sample.
    _, result = run(tmp_path, Editor(), tracks=(Track(default=True),), audio_codec="aac", name="movie.mp4")
    assert (result.status, result.replaced, result.output.name) == ("cleaned", 1, "movie.clean.mp4")
    samples = decode(result.output)
    assert tone_gain(samples, 2.2, window=0.1) < 0.01
    assert tone_gain(samples, 2.2, frequency=NEW_WORD_HZ, window=0.1) == pytest.approx(1.0, abs=0.02)


def test_a_word_that_fails_the_check_is_muted(tmp_path: Path) -> None:
    _, result = run(tmp_path, Editor(works=False))  # the voice model says the same word again
    assert (result.status, result.replaced) == ("cleaned", 0)
    samples = decode(result.output)
    assert tone_gain(samples, 2.2) < 0.01 and tone_gain(samples, 2.2, frequency=NEW_WORD_HZ) < 0.01
    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    [replacement] = report["replacements"]
    assert replacement["replaced"] is False
    assert replacement["reason"] == "a listed word is still heard: 'damn'"
    assert report["output"]["verified_spans"] == 2
    assert "[muted] damn" in (tmp_path / "movie.clean.review.srt").read_text("utf-8")


def test_replacement_waits_for_a_fitting_substitute(tmp_path: Path) -> None:
    editor = Editor()
    _, result = run(tmp_path, editor, **{"replace.substitutes": {"damn": []}})  # a term set to always mute
    assert result.replaced == 0 and editor.texts == []
    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    assert report["replacements"] == []
    assert report["detections"][0]["context"]["substitute_reason"] == "no substitute for this term"
    assert tone_gain(decode(result.output), 2.2) < 0.01


class BrokenEditor(Editor):
    def edit(self, voice: Any, rate: int, text: str, span: tuple[float, float]) -> Any:
        raise RuntimeError("CUDA out of memory")


def test_a_voice_model_that_fails_on_a_word_mutes_it(tmp_path: Path) -> None:
    _, result = run(tmp_path, BrokenEditor())
    assert (result.status, result.replaced) == ("cleaned", 0)
    assert tone_gain(decode(result.output), 2.2) < 0.01
    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    [replacement] = report["replacements"]
    assert replacement["reason"] == "voice replacement failed: RuntimeError: CUDA out of memory"


def test_a_replaced_word_that_fails_in_the_output_is_muted_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The replacement passed its check before rendering; the output is heard again, where a renderer
    # that put the change in the wrong place would let the old word through.
    heard: list[tuple[float, float, float]] = []

    def old_word_heard(self: Pipeline, audio: Any, start: float, span: Any) -> str:
        heard.append((start, span.start, span.end))
        return "a listed word is heard in the output: 'damn'"

    monkeypatch.setattr(Pipeline, "_check_splice", old_word_heard)
    _, result = run(tmp_path, Editor())
    assert heard == [(1.38, 1.88, 2.6)]  # the span, and half a second either side
    assert (result.status, result.replaced) == ("cleaned", 0)
    samples = decode(result.output)
    assert tone_gain(samples, 2.2) < 0.01 and tone_gain(samples, 2.2, frequency=NEW_WORD_HZ) < 0.01
    report = json.loads((tmp_path / "movie.clean.vbr.json").read_text("utf-8"))
    [replacement] = report["replacements"]
    assert (replacement["replaced"], replacement["reason"]) == (
        False,
        "a listed word is heard in the output: 'damn'",
    )
    assert report["output"]["verified_spans"] == 2  # muted, and checked, in both streams
    assert "[muted] damn" in (tmp_path / "movie.clean.review.srt").read_text("utf-8")


def test_windows_past_the_end_of_the_stream_are_cut_short(tmp_path: Path) -> None:
    source = make_clip(tmp_path / "short.mkv", duration=3.0, audio_codec="flac")
    ff = FFmpeg()
    stream = select_audio_stream(probe(ff, source), "en")
    whole, cut, beyond = read_pcm_windows(ff, source, stream, [(1.0, 2.0), (2.5, 4.0), (5.0, 6.0)])
    assert whole.shape == (1, 48_000) and cut.shape == (1, 24_000) and beyond.shape == (1, 0)
    assert tone_gain(whole[0], 0.5, window=0.1) == pytest.approx(1.0, abs=0.02)  # read, not left silent


class Residency:
    """Which stand-in models are alive, and which were alive together each time one loaded."""

    def __init__(self) -> None:
        self.alive: set[str] = set()
        self.together: list[set[str]] = []

    def track(self, name: str, model: Any) -> Any:
        self.alive.add(name)
        self.together.append(set(self.alive))
        weakref.finalize(model, self.alive.discard, name)
        return model


@pytest.mark.parametrize(
    ("keep_loaded", "together"),
    [
        # Transcription, context, separation, the voice model, then Whisper again for the checks.
        (False, [{"whisper"}, {"context"}, {"separator"}, {"editor"}, {"whisper"}]),
        (True, [{"whisper"}, {"whisper", "context"}, {"whisper", "context", "separator"},
                {"whisper", "context", "separator", "editor"}]),
    ],
)  # fmt: skip
def test_large_models_are_loaded_one_at_a_time(
    tmp_path: Path, keep_loaded: bool, together: list[set[str]]
) -> None:
    residency = Residency()
    loaded = load_config(
        None,
        env={},
        cwd=tmp_path,
        overrides={
            "transcription.device": "cpu",
            "replace.enabled": True,
            "context.judge": "",
            "models.keep_loaded": keep_loaded,
        },
    )
    pipeline = Pipeline(
        loaded,
        ui=StrictUI(),
        transcriber_factory=lambda choice: residency.track("whisper", Hearing(SPOKEN)),
        context_models=(
            lambda name, device: residency.track("context", Classifier()),
            lambda name, device: None,
        ),
        voice_models=(
            lambda: residency.track("separator", Separator()),
            lambda: residency.track("editor", Editor()),
            Encoder,
        ),
    )
    source = make_clip(tmp_path / "movie.mkv", audio_codec="flac", tracks=(Track(default=True),))
    result = pipeline.process(source, RunOptions())
    assert (result.status, result.replaced) == ("cleaned", 1)
    assert residency.together == together
