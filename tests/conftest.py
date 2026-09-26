import os

import pytest

from helpers import HAVE_FFMPEG


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "ffmpeg" in item.keywords and not HAVE_FFMPEG:
            item.add_marker(pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH"))
        if "asr" in item.keywords and os.environ.get("VBR_RUN_ASR_TESTS") != "1":
            item.add_marker(pytest.mark.skip(reason="set VBR_RUN_ASR_TESTS=1 to run Whisper tests"))


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read the developer's own config files or environment during tests."""
    monkeypatch.delenv("VBR_CONFIG", raising=False)
    home = tmp_path_factory.mktemp("config-home")
    monkeypatch.setattr(
        "video_beep_remover.config.loader.user_config_path",
        lambda: home / "video-beep-remover" / "config.toml",
    )
