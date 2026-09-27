from pathlib import Path

import pytest

from video_beep_remover.asr.base import build_prompt
from video_beep_remover.asr.faster_whisper import resolve_anchor_model, resolve_model
from video_beep_remover.config.schema import Category, LexiconConfig, TranscriptionConfig
from video_beep_remover.detect.lexicon import compile_lexicon
from video_beep_remover.errors import ConfigError
from video_beep_remover.outputs import resolve_output

MOVIE = Path("/videos/The Movie (2019).mkv")


def test_output_follows_the_template_next_to_the_input() -> None:
    assert resolve_output(MOVIE, "{stem}.clean{ext}", None, many=False) == Path(
        "/videos/The Movie (2019).clean.mkv"
    )


def test_explicit_output_file_or_directory(tmp_path: Path) -> None:
    assert resolve_output(MOVIE, "{stem}.clean{ext}", tmp_path / "x.mkv", many=False) == tmp_path / "x.mkv"
    assert (
        resolve_output(MOVIE, "{stem}.clean{ext}", tmp_path, many=False)
        == tmp_path / "The Movie (2019).clean.mkv"
    )
    assert resolve_output(MOVIE, "{stem}{ext}", tmp_path / "new", many=True) == tmp_path / "new" / MOVIE.name


def test_output_path_without_extension_is_a_new_directory(tmp_path: Path) -> None:
    target = tmp_path / "not-yet-created"
    assert (
        resolve_output(MOVIE, "{stem}.clean{ext}", target, many=False)
        == target / "The Movie (2019).clean.mkv"
    )


def test_absolute_templates_and_bad_placeholders() -> None:
    assert resolve_output(MOVIE, "/out/{stem}{ext}", None, many=False) == Path("/out/The Movie (2019).mkv")
    with pytest.raises(ConfigError, match="placeholder"):
        resolve_output(MOVIE, "{name}{ext}", None, many=False)


def test_prompt_settings() -> None:
    lexicon = compile_lexicon(
        LexiconConfig(
            categories={
                "strong": Category(terms=["*fuck*", "son of a bitch", "re:^f+", "*shit*", "christ's"]),
                "off": Category(enabled=False, terms=["damn"]),
            }
        )
    )
    assert build_prompt("", lexicon) is None
    assert build_prompt("Custom words.", lexicon) == "Custom words."
    assert build_prompt("auto", lexicon) == "Fuck, shit."


def test_auto_model_depends_on_device_strategy_and_language(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = TranscriptionConfig(device="cpu")
    assert resolve_model(cpu, strategy="full", language="en").name == "small.en"
    assert resolve_model(cpu, strategy="full", language="de").name == "small"
    assert resolve_model(cpu, strategy="targeted", language="en").name == "large-v3-turbo"
    choice = resolve_model(cpu, strategy="full", language="en")
    assert (choice.compute_type, choice.batched) == ("int8", False)

    monkeypatch.setattr("video_beep_remover.asr.faster_whisper.cuda_available", lambda: True)
    gpu = resolve_model(TranscriptionConfig(), strategy="full", language="en")
    assert (gpu.name, gpu.device, gpu.compute_type, gpu.batched) == (
        "large-v3-turbo",
        "cuda",
        "float16",
        True,
    )

    explicit = TranscriptionConfig(device="cpu", model="medium", compute_type="float32")
    assert resolve_model(explicit, strategy="full", language="en").describe() == "medium (cpu, float32)"


def test_anchor_model_is_small_and_matches_the_language() -> None:
    cpu = TranscriptionConfig(device="cpu")
    assert resolve_anchor_model(cpu, language="en").describe() == "base.en (cpu, int8)"
    assert resolve_anchor_model(cpu, language="de").name == "base"
    assert not resolve_anchor_model(TranscriptionConfig(device="cuda"), language="en").batched
