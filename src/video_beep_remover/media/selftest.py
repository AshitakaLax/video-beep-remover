"""Render a synthetic tone through the real mute graph (used by `vbr doctor`, DESIGN.md §6.11)."""

import tempfile
from pathlib import Path

import numpy as np

from video_beep_remover.config.schema import OutputConfig
from video_beep_remover.errors import VbrError
from video_beep_remover.media.ffmpeg import FFmpeg, file_arg
from video_beep_remover.media.probe import probe
from video_beep_remover.media.render import StreamAction, StreamPlan, level_dbfs, render
from video_beep_remover.models import CensorInterval

_SPAN = CensorInterval(0.5, 1.0)


def mute_self_test(ff: FFmpeg) -> str | None:
    """None when the mute works; otherwise what went wrong."""
    with tempfile.TemporaryDirectory(prefix="vbr-selftest-") as tmp:
        workdir = Path(tmp)
        source = workdir / "tone.mka"
        try:
            ff.run(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=1.5",
                    "-c:a",
                    "flac",
                    file_arg(source),
                ]
            )
            info = probe(ff, source)
            plan = StreamPlan((StreamAction(info.audio_streams[0], "censor"),), ())
            muted = workdir / "muted.mka"
            render(
                ff,
                info,
                plan,
                [_SPAN],
                output=muted,
                fade=0.01,
                output_config=OutputConfig(),
                workdir=workdir,
            )
            before = np.frombuffer(
                ff.capture(
                    ["-ss", "0.1", "-t", "0.3", "-i", file_arg(muted), "-ac", "1", "-f", "f32le", "pipe:1"]
                ),
                dtype=np.float32,
            )
        except VbrError as exc:
            return str(exc)
        if before.size == 0 or level_dbfs(before) < -40:
            return "audio outside the muted span was lost"
    return None
