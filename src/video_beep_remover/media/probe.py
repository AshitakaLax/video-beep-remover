"""Describe a media file with ffprobe and pick the dialogue audio stream (DESIGN.md §6.1)."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from video_beep_remover.errors import MediaError, UsageError
from video_beep_remover.media.ffmpeg import FFmpeg

# ISO 639-1 codes (as used in the config) and the ISO 639-2 tags containers use.
_ISO639: dict[str, tuple[str, ...]] = {
    "ar": ("ara",), "cs": ("ces", "cze"), "da": ("dan",), "de": ("deu", "ger"), "el": ("ell", "gre"),
    "en": ("eng",), "es": ("spa",), "fi": ("fin",), "fr": ("fra", "fre"), "he": ("heb",),
    "hi": ("hin",), "hu": ("hun",), "id": ("ind",), "it": ("ita",), "ja": ("jpn",), "ko": ("kor",),
    "nl": ("nld", "dut"), "no": ("nor", "nob", "nno"), "pl": ("pol",), "pt": ("por",),
    "ro": ("ron", "rum"), "ru": ("rus",), "sv": ("swe",), "th": ("tha",), "tr": ("tur",),
    "uk": ("ukr",), "vi": ("vie",), "zh": ("zho", "chi"),
}  # fmt: skip
_COMMENTARY = re.compile(r"comment", re.IGNORECASE)
_DESCRIPTION = re.compile(r"descri", re.IGNORECASE)

StreamKind = Literal["video", "audio", "subtitle", "data", "attachment", "unknown"]


def lang_matches(tag: str | None, code: str) -> bool:
    """Compare a container language tag ("eng") with a config language code ("en")."""
    if not tag:
        return False
    tag_base = tag.lower().split("-")[0]
    code_base = code.lower().split("-")[0]
    if tag_base == code_base:
        return True
    return tag_base in _ISO639.get(code_base, ()) or code_base in _ISO639.get(tag_base, ())


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class StreamInfo:
    index: int
    kind: StreamKind
    codec: str | None = None
    language: str | None = None
    title: str | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_rate: int | None = None
    bit_rate: int | None = None
    start_time: float | None = None
    disposition: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return bool(self.disposition.get("default"))

    @property
    def is_commentary(self) -> bool:
        return bool(self.disposition.get("comment")) or bool(self.title and _COMMENTARY.search(self.title))

    @property
    def is_audio_description(self) -> bool:
        return bool(self.disposition.get("visual_impaired")) or bool(
            self.title and _DESCRIPTION.search(self.title)
        )

    @property
    def is_attached_picture(self) -> bool:
        return bool(self.disposition.get("attached_pic"))

    def describe(self) -> str:
        parts = [f"#{self.index} {self.codec or '?'}"]
        if self.kind == "audio" and self.channels:
            parts.append(self.channel_layout or f"{self.channels}ch")
        if self.language:
            parts.append(self.language)
        if self.title:
            parts.append(repr(self.title))
        return " ".join(parts)


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    format_name: str
    duration: float
    start_time: float
    size: int
    streams: tuple[StreamInfo, ...]

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.kind == "audio"]

    def stream(self, index: int) -> StreamInfo:
        for stream in self.streams:
            if stream.index == index:
                return stream
        raise UsageError(f"{self.path.name} has no stream #{index}")


def parse_probe(path: Path, data: Mapping[str, Any]) -> MediaInfo:
    fmt = data.get("format") or {}
    streams = []
    for raw in data.get("streams") or []:
        tags = {str(k).lower(): str(v) for k, v in (raw.get("tags") or {}).items()}
        kind = raw.get("codec_type")
        language = tags.get("language")
        streams.append(
            StreamInfo(
                index=int(raw["index"]),
                kind=kind if kind in ("video", "audio", "subtitle", "data", "attachment") else "unknown",
                codec=raw.get("codec_name"),
                language=None if language in (None, "", "und") else language,
                title=tags.get("title") or None,
                channels=_int(raw.get("channels")),
                channel_layout=raw.get("channel_layout"),
                sample_rate=_int(raw.get("sample_rate")),
                bit_rate=_int(raw.get("bit_rate")),
                start_time=_float(raw.get("start_time")),
                disposition={str(k): int(v) for k, v in (raw.get("disposition") or {}).items()},
            )
        )
    duration = _float(fmt.get("duration"))
    if duration is None:
        durations = [_float(raw.get("duration")) for raw in data.get("streams") or []]
        known = [d for d in durations if d is not None]
        duration = max(known) if known else None
    if duration is None or duration <= 0:
        raise MediaError(f"could not determine the duration of {path}")
    return MediaInfo(
        path=path,
        format_name=str(fmt.get("format_name") or ""),
        duration=duration,
        start_time=_float(fmt.get("start_time")) or 0.0,
        size=_int(fmt.get("size")) or 0,
        streams=tuple(streams),
    )


def probe(ff: FFmpeg, path: Path) -> MediaInfo:
    return parse_probe(path, ff.probe(path))


def select_audio_stream(info: MediaInfo, language: str, requested: int | str = "auto") -> StreamInfo:
    """Pick the dialogue track: skip commentary and audio description, then prefer the configured
    language, the default flag, more channels and a lower index."""
    audio = info.audio_streams
    if not audio:
        raise MediaError(f"{info.path.name} has no audio stream")
    if requested != "auto":
        stream = info.stream(int(requested))
        if stream.kind != "audio":
            raise UsageError(f"stream #{requested} of {info.path.name} is {stream.kind}, not audio")
        return stream
    candidates = [s for s in audio if not s.is_commentary and not s.is_audio_description] or audio
    return min(
        candidates,
        key=lambda s: (not lang_matches(s.language, language), not s.is_default, -(s.channels or 0), s.index),
    )
