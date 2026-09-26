"""Real Whisper on synthesized speech. Opt in with VBR_RUN_ASR_TESTS=1 (downloads a model)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from helpers import run_ffmpeg, srt
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


FILM = [
    ("We found the copper lantern near the old harbor.", None),
    ("What the hell is going on over there?", None),
    ("Every pilgrim needs a marble saddle for the journey.", None),
    ("Get the fuck out of my orchard right now.", "Get the f*** out of my orchard right now."),
    ("The glacier violin sounds like distant thunder again.", None),
    ("You lying bastard, I knew it all along.", ""),  # spoken, but not in the subtitles
    ("Meadow falcons never sleep before the dawn arrives.", None),
    ("Honestly the weather here is lovely in spring.", None),
]


def speech_film(folder: Path) -> Path:
    """Sentences with known timing, a video track, and verbatim subtitles (one line missing)."""
    import wave

    rate, gap, t = 22050, 1.5, 1.0
    pcm = b"\x00\x00" * int(rate * t)
    cues = []
    for i, (spoken, shown) in enumerate(FILM):
        path = folder / f"line{i}.wav"
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w", str(path), spoken], check=True)
        with wave.open(str(path)) as line:
            frames = line.readframes(line.getnframes())
        length = len(frames) / 2 / rate
        if shown != "":
            cues.append((t - 0.1, t + length + 0.3, shown or spoken))
        pcm += frames + b"\x00\x00" * int(rate * gap)
        t += length + gap
    speech = folder / "speech.wav"
    with wave.open(str(speech), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    video = folder / "film.mkv"
    run_ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24", "-i", str(speech), "-shortest",
               "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video))  # fmt: skip
    (folder / "film.en.srt").write_text(srt(*cues), "utf-8")
    return video


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="needs espeak-ng to synthesize speech")
def test_subtitle_guided_strategies_find_what_full_transcription_finds(tmp_path: Path) -> None:
    """The M2 criterion: with verbatim subtitles, targeted mode finds what full mode finds."""
    video = speech_film(tmp_path)
    # A dense test film: three flagged lines in 45 s would otherwise trip the coverage guard.
    overrides = {
        "transcription.device": "cpu",
        "transcription.model": "small.en",
        "analysis.targeted.max_coverage": 0.9,
    }

    def scan(strategy: str) -> dict[str, dict[str, float]]:
        loaded = load_config(
            None, env={}, cwd=tmp_path, overrides=overrides | {"analysis.strategy": strategy}
        )
        result = Pipeline(loaded).process(
            video, RunOptions(dry_run=True, report=tmp_path / f"{strategy}.json")
        )
        assert result.strategy == strategy
        report = json.loads((tmp_path / f"{strategy}.json").read_text("utf-8"))
        return {d["heard"].strip(",.!?").lower(): d for d in report["detections"]}

    full, targeted, hybrid = scan("full"), scan("targeted"), scan("hybrid")
    assert {"hell", "fuck", "bastard"} <= set(full)
    assert {"hell", "fuck"} <= set(targeted) and "bastard" not in targeted  # the unsubtitled line
    assert {"hell", "fuck", "bastard"} <= set(hybrid)
    for word in ("hell", "fuck"):
        assert targeted[word]["source"] == "asr"
        assert abs(targeted[word]["start"] - full[word]["start"]) < 0.1
        assert abs(targeted[word]["end"] - full[word]["end"]) < 0.1
