"""The real context and voice models, on made-up text and audio. The stand-ins the other tests use cannot
catch what goes wrong inside a model, such as tensors left on the wrong device: F5-TTS once failed on
every GPU that way, and SpeechBrain on Windows, which needs Developer Mode for its symlinks.

Opt in with VBR_RUN_MODEL_TESTS=1 (the [context] and [voice] extras; the models download on first use,
and a GPU makes them fast). VBR_TEST_JUDGE names a judge model to try as well, e.g. Qwen/Qwen3-1.7B."""

import os
from pathlib import Path

import numpy as np
import pytest

from video_beep_remover.config.schema import ContextConfig, ReplaceConfig
from video_beep_remover.context.analyse import parse_answer, sense_question
from video_beep_remover.context.lines import Line
from video_beep_remover.context.models import LocalJudge, ToxicityClassifier, torch_device
from video_beep_remover.voice.models import DemucsSeparator, EcapaEncoder, F5Editor

pytestmark = pytest.mark.models

RATE = 24_000


@pytest.fixture(scope="module")
def device() -> str:
    return torch_device("auto")


@pytest.fixture(scope="module")
def voice() -> np.ndarray:
    """Three seconds of quiet noise: enough for every model to run its whole way through."""
    return (np.random.default_rng(1).standard_normal(3 * RATE) * 0.05).astype(np.float32)


def test_the_classifier_scores_a_rude_line_above_a_clean_one(device: str) -> None:
    classifier = ToxicityClassifier(ContextConfig().classifier, device=device, offline=False)
    rude, clean = classifier.score(["You stupid bastard, I will kill you.", "The weather is lovely today."])
    assert all(0.0 <= value <= 1.0 for value in (*rude.values(), *clean.values()))
    assert rude["toxicity"] > 0.5 > clean["toxicity"]


@pytest.mark.skipif(not os.environ.get("VBR_TEST_JUDGE"), reason="set VBR_TEST_JUDGE to a judge model")
def test_a_judge_answers_with_json(device: str) -> None:
    judge = LocalJudge(os.environ["VBR_TEST_JUDGE"], device=device, offline=False)
    lines = [Line(0.0, 2.0, "The farmer loaded his ass with firewood.", cue=1)]
    assert parse_answer(judge.ask(sense_question(lines, 0, "ass"))).get("use") in ("profane", "harmless")


def test_the_separation_model_keeps_the_shape(device: str, voice: np.ndarray) -> None:
    separator = DemucsSeparator(ReplaceConfig().separation, device=device, offline=False)
    vocals = separator.vocals(np.stack([voice, voice]), RATE)
    assert vocals.shape == (2, voice.size) and vocals.dtype == np.float32


def test_the_voice_model_says_a_span_again(device: str, voice: np.ndarray) -> None:
    editor = F5Editor(ReplaceConfig().model, device=device, offline=False, steps=4)
    edited = editor.edit(voice, RATE, "What the heck was that?", (1.0, 1.4))
    assert edited.shape == voice.shape and np.isfinite(edited).all()


def test_the_speaker_encoder_embeds_a_voice(device: str, voice: np.ndarray, tmp_path: Path) -> None:
    embedding = EcapaEncoder(tmp_path, device=device, offline=False).embed(voice, RATE)
    assert embedding.ndim == 1 and embedding.size >= 64 and np.isfinite(embedding).all()
