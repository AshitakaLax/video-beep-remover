"""The renderer against real FFmpeg, measuring pure tones in the output."""

from pathlib import Path

import pytest

from helpers import Track, decode, make_clip, tone_gain
from video_beep_remover.config.schema import OutputConfig
from video_beep_remover.errors import RenderError
from video_beep_remover.media import render as render_module
from video_beep_remover.media.audio import SAMPLE_RATE, decode_track, read_window
from video_beep_remover.media.ffmpeg import FFmpeg
from video_beep_remover.media.probe import probe, select_audio_stream
from video_beep_remover.media.render import plan_streams, render
from video_beep_remover.media.selftest import mute_self_test
from video_beep_remover.models import CensorInterval as Span

pytestmark = pytest.mark.ffmpeg


@pytest.fixture(scope="module")
def ff() -> FFmpeg:
    return FFmpeg()


def clean(ff: FFmpeg, source: Path, spans: list[Span], tmp_path: Path, **output: object) -> Path:
    info = probe(ff, source)
    analysed = select_audio_stream(info, "en")
    config = OutputConfig(**output)  # type: ignore[arg-type]
    plan = plan_streams(info, analysed, config, "en")
    target = tmp_path / f"out{source.suffix}"
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    render(ff, info, plan, spans, output=target, fade=0.01, output_config=config, workdir=workdir)
    return target


@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
def test_spans_are_muted_with_fades_on_the_requested_samples(ff: FFmpeg, tmp_path: Path, suffix: str) -> None:
    # Edges fall mid-cycle of the 440 Hz tone, where a hard cut would click.
    source = make_clip(tmp_path / f"in{suffix}", audio_codec="flac" if suffix == ".mkv" else "aac")
    out = clean(ff, source, [Span(2.0006, 2.5011), Span(4.1003, 4.6007)], tmp_path)
    samples = decode(out)
    assert tone_gain(samples, 1.0) == pytest.approx(1.0, abs=0.1)
    assert tone_gain(samples, 2.25) < 0.01
    assert tone_gain(samples, 4.35) < 0.01
    assert tone_gain(samples, 3.3) == pytest.approx(1.0, abs=0.1)
    assert 0.2 < tone_gain(samples, 2.0056, window=0.002) < 0.95  # halfway through the fade-out
    assert tone_gain(samples, 2.0126, window=0.002) < 0.05  # fade complete after 10 ms


def test_encoder_priming_that_moves_the_output_timeline_is_measured(ff: FFmpeg, tmp_path: Path) -> None:
    """AAC at 22.05 kHz in Matroska starts 46 ms before zero, and re-encoding moves everything 46 ms
    later on the output's timeline. The mutes still hit the right samples, and verification has to
    look 46 ms later too."""
    source = make_clip(tmp_path / "in.mkv", sample_rate=22_050)
    info = probe(ff, source)
    assert info.start_time == pytest.approx(-0.046, abs=0.001)
    analysed = select_audio_stream(info, "en")
    config = OutputConfig()
    plan = plan_streams(info, analysed, config, "en")
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = render(
        ff,
        info,
        plan,
        [Span(2.0, 2.5)],
        output=tmp_path / "out.mkv",
        fade=0.01,
        output_config=config,
        workdir=workdir,
    )
    shift = result.timeline_shift
    assert shift == pytest.approx(0.046, abs=0.002)
    samples = decode(result.output)
    assert tone_gain(samples, 1.9 + shift) == pytest.approx(1.0, abs=0.1)
    assert tone_gain(samples, 2.02 + shift) < 0.01
    assert tone_gain(samples, 2.48 + shift) < 0.01
    assert tone_gain(samples, 2.6 + shift) == pytest.approx(1.0, abs=0.1)


def test_late_starting_audio_keeps_mutes_on_the_media_timeline(ff: FFmpeg, tmp_path: Path) -> None:
    source = make_clip(tmp_path / "late.mkv", tracks=[Track(default=True, delay=0.5)], audio_codec="flac")
    track = probe(ff, source).audio_streams[0]
    assert (track.start_time or 0) > 0.4

    audio = decode_track(ff, source, track.index, tmp_path / "track.f32")
    onset = next(i for i, v in enumerate(audio) if abs(v) > 0.01) / SAMPLE_RATE
    assert onset == pytest.approx(0.5, abs=0.01)  # sample 0 is media time 0, not the first audio sample

    window = read_window(ff, source, track.index, 3.0, 1.0)
    assert len(window) == SAMPLE_RATE

    samples = decode(clean(ff, source, [Span(3.0, 3.6)], tmp_path))
    assert tone_gain(samples, 3.3) < 0.01
    assert tone_gain(samples, 2.9) == pytest.approx(1.0, abs=0.1)
    assert tone_gain(samples, 3.7) == pytest.approx(1.0, abs=0.1)


def test_same_language_mix_is_muted_and_commentary_dropped(ff: FFmpeg, tmp_path: Path) -> None:
    tracks = [Track(440, default=True), Track(660), Track(880, title="Director commentary")]
    source = make_clip(tmp_path / "multi.mkv", tracks=tracks, audio_codec="flac")
    out = clean(ff, source, [Span(2.0, 2.5)], tmp_path)
    audio_streams = probe(ff, out).audio_streams
    assert len(audio_streams) == 2
    assert tone_gain(decode(out, "0:a:0"), 2.25) < 0.01
    assert tone_gain(decode(out, "0:a:1"), 2.25, frequency=660) < 0.01
    assert tone_gain(decode(out, "0:a:1"), 1.0, frequency=660) == pytest.approx(1.0, abs=0.1)


def test_video_is_copied_and_tags_survive(ff: FFmpeg, tmp_path: Path) -> None:
    # MKV, because MP4 keeps no per-stream title.
    source = make_clip(tmp_path / "in.mkv", tracks=[Track(language="eng", title="Main", default=True)])
    out = clean(ff, source, [Span(1.0, 1.5)], tmp_path)
    before, after = probe(ff, source), probe(ff, out)
    assert [s.codec for s in after.streams] == [s.codec for s in before.streams]
    audio = after.audio_streams[0]
    assert (audio.language, audio.title, audio.is_default) == ("eng", "Main", True)


def test_subtitles_are_dropped_by_default_and_kept_when_asked(ff: FFmpeg, tmp_path: Path) -> None:
    source = make_clip(tmp_path / "subs.mkv", subtitles="1\n00:00:01,000 --> 00:00:02,000\nWhat the hell?\n")
    assert [s.kind for s in probe(ff, clean(ff, source, [Span(1.0, 1.5)], tmp_path)).streams] == [
        "video",
        "audio",
    ]
    kept = clean(ff, source, [Span(1.0, 1.5)], tmp_path, subtitle_streams="copy")
    assert [s.kind for s in probe(ff, kept).streams] == ["video", "audio", "subtitle"]


def test_a_rejected_command_fails_verification_and_leaves_no_output(
    ff: FFmpeg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(spans: list[Span], label: str, fade: float) -> tuple[str, str]:
        # Commands addressed to a filter that doesn't exist are dropped by FFmpeg without an error.
        text, initial = original(spans, label, fade)
        return text.replace(f"afade@{label} ", "afade@missing "), initial

    original = render_module.command_file
    monkeypatch.setattr(render_module, "command_file", broken)
    source = make_clip(tmp_path / "in.mkv", audio_codec="flac")
    with pytest.raises(RenderError, match="verification failed"):
        clean(ff, source, [Span(2.0, 2.5)], tmp_path)
    assert not (tmp_path / "out.mkv").exists()
    assert not (tmp_path / "out.partial.mkv").exists()


def test_self_test_passes(ff: FFmpeg) -> None:
    assert mute_self_test(ff) is None
