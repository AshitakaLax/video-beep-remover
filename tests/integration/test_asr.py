"""Real Whisper on synthesized speech. Opt in with VBR_RUN_ASR_TESTS=1 (downloads a model)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from helpers import run_ffmpeg
from video_beep_remover.config import load_config
from video_beep_remover.pipeline import Pipeline, RunOptions

pytestmark = [pytest.mark.ffmpeg, pytest.mark.asr]


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="needs espeak-ng to synthesize speech")
def test_whisper_finds_and_mutes_spoken_words(tmp_path: Path) -> None:
    speech = tmp_path / "speech.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "135", "-w", str(speech),
         "Hello there. This is a damn good example of speech. What the hell was that noise?"],
        check=True,
    )  # fmt: skip
    video = tmp_path / "speech.mp4"
    run_ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24", "-i", str(speech), "-shortest",
               "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video))  # fmt: skip

    loaded = load_config(None, env={}, cwd=tmp_path, overrides={"transcription.device": "cpu"})
    result = Pipeline(loaded).process(video, RunOptions())
    assert result.status == "cleaned"
    assert result.detections == 2

    from faster_whisper import WhisperModel

    assert result.output is not None
    segments, _ = WhisperModel("small.en", device="cpu", compute_type="int8").transcribe(str(result.output))
    text = " ".join(segment.text for segment in segments).lower()
    assert not re.search(r"\b(damn|hell)\b", text), text
    assert "example of speech" in text
