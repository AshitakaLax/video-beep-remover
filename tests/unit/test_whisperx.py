"""The whisperx backend against a stand-in for WhisperX: no PyTorch or models needed."""

import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from helpers import StrictUI
from video_beep_remover.asr.base import Clip
from video_beep_remover.asr.faster_whisper import ModelChoice, Segment
from video_beep_remover.asr.whisperx import MARGIN_S, Aligner, WhisperXTranscriber, status
from video_beep_remover.config import load_config
from video_beep_remover.errors import ConfigError, DependencyError
from video_beep_remover.media.audio import SAMPLE_RATE
from video_beep_remover.models import Word
from video_beep_remover.pipeline import Pipeline, _CacheScope

CLIP = Clip(100.0, np.zeros(10 * SAMPLE_RATE, dtype=np.float32))  # 100-110 s on the media timeline
WEIGHTS = "wav2vec2_fairseq_base_ls960_asr_ls960.pth"


class FakeWhisperX(types.ModuleType):
    """WhisperX's load_align_model and align. `times` places words (or characters, for unspaced
    languages) relative to the audio align is given; a word missing from it is left unplaced."""

    def __init__(self, hub: Path) -> None:
        super().__init__("whisperx")
        self.alignment = types.ModuleType("whisperx.alignment")
        self.alignment.DEFAULT_ALIGN_MODELS_TORCH = {"en": "WAV2VEC2_ASR_BASE_960H"}  # type: ignore[attr-defined]
        self.alignment.DEFAULT_ALIGN_MODELS_HF = {"ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"}  # type: ignore[attr-defined]
        self.hub = hub
        self.loaded: list[tuple[str, str, str | None, bool]] = []
        self.calls: list[tuple[dict[str, Any], int]] = []
        self.times: dict[str, tuple[float, float]] = {}
        self.drop = False  # lose a word, as when a segment fails to align
        self.fail = False
        self.punkt = True  # NLTK's punkt_tab is downloaded
        self.downloads = 0

    def load_align_model(
        self, language_code: str, device: str, model_name: str | None = None, model_cache_only: bool = False
    ) -> tuple[str, dict[str, str]]:
        self.loaded.append((language_code, device, model_name, model_cache_only))
        return "model", {"language": language_code}

    def align(
        self, transcript: list[dict[str, Any]], model: str, metadata: dict[str, str], audio: Any, device: str
    ) -> dict[str, Any]:
        [segment] = transcript
        self.calls.append((segment, len(audio)))
        if self.fail:
            raise RuntimeError("backtrack failed")
        text = segment["text"]
        units = list(text) if metadata["language"] in ("ja", "zh") else text.split(" ")
        found = [{"word": unit, **self._placed(unit)} for unit in units]
        if self.drop:
            found.pop()
        return {"segments": [{"words": found}], "word_segments": found}

    def _placed(self, unit: str) -> dict[str, float]:
        if unit not in self.times:
            return {}
        start, end = self.times[unit]
        return {"start": start, "end": end, "score": 0.9}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeWhisperX:
    module = FakeWhisperX(tmp_path / "hub" / "checkpoints")
    monkeypatch.setitem(sys.modules, "whisperx", module)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", module.alignment)

    def find(resource: str) -> str:
        if not module.punkt:
            raise LookupError(resource)
        return resource

    def download(package: str, quiet: bool = False) -> bool:
        module.downloads += 1
        module.punkt = True
        return True

    nltk = types.ModuleType("nltk")
    nltk.data = SimpleNamespace(find=find)  # type: ignore[attr-defined]
    nltk.download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    torch = types.ModuleType("torch")
    torch.hub = SimpleNamespace(get_dir=lambda: str(tmp_path / "hub"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.pipelines = SimpleNamespace(  # type: ignore[attr-defined]
        __all__=["WAV2VEC2_ASR_BASE_960H"], WAV2VEC2_ASR_BASE_960H=SimpleNamespace(_path=WEIGHTS)
    )
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    return module


def segment(*words: Word, start: float, end: float, clip: int = 0) -> Segment:
    return Segment(clip, start, end, words)


def test_words_are_widened_to_their_aligned_times(fake: FakeWhisperX) -> None:
    first = 102.0 - MARGIN_S  # the aligner hears the segment with a margin; its times start there
    fake.times = {
        "What": (102.0 - first, 102.4 - first),  # starts and ends later than Whisper heard it
        "hell?": (102.75 - first, 103.2 - first),  # starts later, ends later
        "it": (103.3 - first, 104.4 - first),  # ends 0.7 s after Whisper's end: a misalignment
    }
    heard = segment(
        Word(" What", 102.1, 102.3, 0.9),
        Word(" the", 102.3, 102.5, 0.8),
        Word(" hell?", 102.6, 102.9, 0.7),
        Word(" it", 103.3, 103.7, 0.6),
        start=102.0,
        end=104.0,
    )
    words = Aligner("en", "auto", "cpu", offline=False).retime(CLIP, heard)
    [(sent, samples)] = fake.calls
    assert sent == {"start": 0.0, "end": pytest.approx(2.0 + 2 * MARGIN_S), "text": "What the hell? it"}
    assert samples == round((2.0 + 2 * MARGIN_S) * SAMPLE_RATE)
    # The earlier start and the later end of the two: a word censored late is heard.
    assert [(w.text, round(w.start, 3), round(w.end, 3), w.probability) for w in words] == [
        (" What", 102.0, 102.4, 0.9),
        (" the", 102.3, 102.5, 0.8),  # not placed by the aligner: Whisper's times
        (" hell?", 102.6, 103.2, 0.7),
        (" it", 103.3, 103.7, 0.6),
    ]


def test_the_audio_given_to_the_aligner_stays_inside_the_clip(fake: FakeWhisperX) -> None:
    heard = segment(Word(" Damn", 100.05, 100.4), Word(" it", 109.5, 109.9), start=100.0, end=109.95)
    Aligner("en", "auto", "cpu", offline=False).retime(CLIP, heard)
    assert fake.calls[0][1] == len(CLIP.audio)


def test_whisper_times_are_kept_when_alignment_fails(
    fake: FakeWhisperX, caplog: pytest.LogCaptureFixture
) -> None:
    heard = segment(Word(" Oh", 102.1, 102.3), Word(" damn", 102.4, 102.6), start=102.0, end=103.0)
    aligner = Aligner("en", "auto", "cpu", offline=False)
    fake.times = {"Oh": (0.1, 0.2), "damn": (0.6, 0.9)}
    fake.drop = True  # a word missing from the result: the words can no longer be paired up
    assert aligner.retime(CLIP, heard) == list(heard.words)
    fake.drop, fake.fail = False, True
    with caplog.at_level(logging.WARNING):
        assert aligner.retime(CLIP, heard) == list(heard.words)
    assert "could not align 'Oh damn'" in caplog.text


def test_blank_words_are_kept_but_not_aligned(fake: FakeWhisperX) -> None:
    fake.times = {"Hell": (0.25, 0.5), "no": (0.7, 0.9)}
    heard = segment(
        Word(" Hell", 102.1, 102.3),
        Word(" ", 102.3, 102.3),
        Word(" no", 102.4, 102.6),
        start=102.0,
        end=103.0,
    )
    words = Aligner("en", "auto", "cpu", offline=False).retime(CLIP, heard)
    assert fake.calls[0][0]["text"] == "Hell no"
    assert words[1] == Word(" ", 102.3, 102.3)
    assert [round(w.start, 3) for w in words] == [
        round(102.0 - MARGIN_S + 0.25, 3),  # earlier than Whisper's start
        102.3,
        102.4,  # Whisper's start, the earlier one
    ]


def test_unspaced_languages_are_aligned_character_by_character(fake: FakeWhisperX) -> None:
    fake.times = {"く": (0.3, 0.4), "そ": (0.4, 0.6), "ば": (0.7, 0.8), "か": (0.8, 1.0)}
    heard = segment(Word("くそ", 102.1, 102.3), Word(" ばか", 102.4, 102.6), start=102.0, end=103.0)
    words = Aligner("ja", "auto", "cpu", offline=False).retime(CLIP, heard)
    assert fake.calls[0][0]["text"] == "くそばか"
    first = 102.0 - MARGIN_S
    # each word from its first character's start to its last one's end, widened as usual
    assert [(round(w.start - first, 3), round(w.end - first, 3)) for w in words] == [(0.3, 0.6), (0.6, 1.0)]
    assert fake.loaded == [("ja", "cpu", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", False)]


def test_the_alignment_model_follows_the_language(fake: FakeWhisperX) -> None:
    Aligner("en", "auto", "cuda", offline=False)
    Aligner("sw", "someone/wav2vec2-swahili", "cpu", offline=False)
    assert fake.loaded == [
        ("en", "cuda", "WAV2VEC2_ASR_BASE_960H", False),
        ("sw", "cpu", "someone/wav2vec2-swahili", False),
    ]
    with pytest.raises(ConfigError, match="no alignment model for language 'sw'"):
        Aligner("sw", "auto", "cpu", offline=False)


def test_offline_the_model_and_nltk_data_must_be_downloaded(
    fake: FakeWhisperX, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(DependencyError, match="WAV2VEC2_ASR_BASE_960H is not downloaded"):
        Aligner("en", "auto", "cpu", offline=True)
    assert status("en", "auto") == (
        False,
        "WAV2VEC2_ASR_BASE_960H; not downloaded yet: WAV2VEC2_ASR_BASE_960H (the first run downloads them)",
    )
    fake.hub.mkdir(parents=True)
    (fake.hub / WEIGHTS).write_bytes(b"weights")
    Aligner("en", "auto", "cpu", offline=True)
    assert fake.loaded[-1] == ("en", "cpu", "WAV2VEC2_ASR_BASE_960H", True)
    assert status("en", "auto") == (True, "WAV2VEC2_ASR_BASE_960H is downloaded")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo, name: None)
    with pytest.raises(DependencyError, match="someone/model is not downloaded"):
        Aligner("sw", "someone/model", "cpu", offline=True)

    fake.punkt = False
    with pytest.raises(DependencyError, match="punkt_tab"):
        Aligner("en", "auto", "cpu", offline=True)
    assert fake.downloads == 0
    Aligner("en", "auto", "cpu", offline=False)  # online: fetched first
    assert fake.downloads == 1


def test_without_whisperx_the_backend_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "whisperx", None)
    with pytest.raises(DependencyError, match=r"video-beep-remover\[align\]"):
        Aligner("en", "auto", "cpu", offline=False)


class FakeWhisper:
    """FasterWhisperTranscriber.segments: two segments in the first clip, one in the second."""

    name = "large-v3-turbo (cpu, int8)"
    choice = ModelChoice("large-v3-turbo", "cpu", "int8", batched=False, align=True)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def segments(
        self, clips: Any, *, language: str, prompt: Any, vad: bool, on_progress: Any
    ) -> Iterator[Segment]:
        for index, (start, end) in enumerate([(100.5, 101.5), (102.0, 103.0), (200.5, 201.0)]):
            clip = 0 if index < 2 else 1
            yield segment(Word(f" w{index}", start + 0.1, start + 0.3), start=start, end=end, clip=clip)
            self.events.append(f"progress {index}")
            on_progress(float(index))


def test_segments_are_aligned_as_they_are_decoded(fake: FakeWhisperX) -> None:
    events: list[str] = []
    original = fake.align

    def align(*args: Any) -> dict[str, Any]:
        events.append(f"align {args[0][0]['text']}")
        return original(*args)

    fake.align = align  # type: ignore[method-assign]
    fake.times = {"w0": (0.4, 0.6), "w2": (0.3, 0.5)}
    transcriber = WhisperXTranscriber(FakeWhisper(events), language="en", align_model="auto", offline=False)  # type: ignore[arg-type]
    assert transcriber.name == "large-v3-turbo (cpu, int8) with alignment"
    assert len(fake.loaded) == 1  # loaded up front, so a missing model fails before transcribing
    clips = [CLIP, Clip(200.0, np.zeros(5 * SAMPLE_RATE, dtype=np.float32))]
    progress: list[float] = []
    [first, second] = transcriber.transcribe(clips, language="en", prompt=None, on_progress=progress.append)
    assert events == ["align w0", "progress 0", "align w1", "progress 1", "align w2", "progress 2"]
    assert progress == [0.0, 1.0, 2.0]
    assert [(w.text, round(w.start, 3), round(w.end, 3)) for w in first] == [
        (" w0", 100.6, round(100.5 - MARGIN_S + 0.6, 3)),  # the aligned end is later than Whisper's
        (" w1", 102.1, 102.3),
    ]
    assert [(w.text, round(w.start, 3), round(w.end, 3)) for w in second] == [(" w2", 200.6, 200.8)]
    assert len(fake.loaded) == 1


def test_the_whisperx_backend_aligns_the_models_that_find_words(tmp_path: Path) -> None:
    def pipeline(**overrides: Any) -> Pipeline:
        loaded = load_config(
            None, env={}, cwd=tmp_path, overrides={"transcription.device": "cpu", **overrides}
        )
        run = Pipeline(loaded, ui=StrictUI(), transcriber_factory=lambda choice: None)  # type: ignore[arg-type,return-value]
        run._scope = _CacheScope("fingerprint", 1, 100.0, None)
        return run

    plain, aligned = pipeline(), pipeline(**{"transcription.backend": "whisperx"})
    assert not plain.model_choice("hybrid").align
    assert aligned.model_choice("hybrid").align and aligned.model_choice("full").align
    assert not aligned.model_choice("anchor").align  # anchors are only matched as text
    assert aligned.model_choice("hybrid").describe() == "large-v3-turbo (cpu, int8) with alignment"

    def key(run: Pipeline, role: str) -> str:
        store = run.transcripts(role)
        assert store is not None
        return store.path.name

    assert key(aligned, "hybrid") != key(plain, "hybrid")  # aligned words are cached apart
    assert key(aligned, "anchor") == key(plain, "anchor")
    other = pipeline(**{"transcription.backend": "whisperx", "transcription.align_model": "someone/model"})
    assert key(other, "hybrid") != key(aligned, "hybrid")
    assert key(pipeline(**{"transcription.align_model": "someone/model"}), "hybrid") == key(plain, "hybrid")


def test_the_pipeline_wraps_faster_whisper_when_aligning(
    fake: FakeWhisperX, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[ModelChoice] = []

    class Whisper:
        def __init__(self, choice: ModelChoice, **settings: Any) -> None:
            built.append(choice)
            self.choice, self.name = choice, choice.describe()

    monkeypatch.setattr("video_beep_remover.pipeline.FasterWhisperTranscriber", Whisper)
    loaded = load_config(
        None,
        env={},
        cwd=tmp_path,
        overrides={
            "transcription.device": "cpu",
            "transcription.backend": "whisperx",
            "analysis.language": "ja",
        },
    )
    run = Pipeline(loaded, ui=StrictUI())
    assert isinstance(run._load_transcriber(run.model_choice("hybrid")), WhisperXTranscriber)
    assert isinstance(run._load_transcriber(run.model_choice("anchor")), Whisper)
    assert fake.loaded == [("ja", "cpu", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", False)]
