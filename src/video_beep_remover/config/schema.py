"""Configuration schema (DESIGN.md §4). The packaged defaults.toml fills in every field."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    # A misspelled key is an error, reported with its TOML path.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Category(_Model):
    enabled: bool = True
    terms: list[str] = Field(default_factory=list)


class Hints(_Model):
    terms: list[str] = Field(default_factory=list)


class LexiconConfig(_Model):
    language: str = "en"
    files: list[str] = Field(default_factory=list)
    allow: list[str] = Field(default_factory=list)
    detect_masked: bool = True
    masked_patterns: list[str] = Field(default_factory=lambda: ["[*#]"])
    categories: dict[str, Category] = Field(default_factory=dict)
    hints: Hints = Field(default_factory=Hints)


class CensorConfig(_Model):
    pad_before_ms: int = Field(120, ge=0, le=2000)
    pad_after_ms: int = Field(200, ge=0, le=2000)
    min_duration_ms: int = Field(250, ge=0, le=5000)
    merge_gap_ms: int = Field(250, ge=0, le=5000)
    fade_ms: int = Field(10, ge=1, le=100)
    refine_edges: bool = False


class TargetedConfig(_Model):
    window_padding_s: float = Field(1.5, ge=0)
    min_window_s: float = Field(4.0, ge=0)
    max_window_s: float = Field(30.0, gt=0, le=30)
    merge_gap_s: float = Field(1.0, ge=0)
    max_coverage: float = Field(0.35, gt=0, le=1)
    on_unconfirmed: Literal["estimate", "cue", "skip"] = "estimate"
    expand_by_s: float = Field(3.0, ge=0)


class SyncConfig(_Model):
    anchors: int = Field(6, ge=0)
    trusted_search_s: float = Field(3.0, ge=0)
    untrusted_search_s: float = Field(20.0, ge=0)
    min_matched_ratio: float = Field(0.5, ge=0, le=1)
    max_error_s: float = Field(0.5, ge=0)
    min_fidelity: float = Field(0.6, ge=0, le=1)
    ffsubsync: Literal["never", "fallback"] = "fallback"


Strategy = Literal["hybrid", "targeted", "full"]


class AnalysisConfig(_Model):
    strategy: Strategy = "hybrid"
    fallback_to_full: bool = True
    audio_stream: Literal["auto"] | Annotated[int, Field(ge=0)] = "auto"
    language: str = "en"
    targeted: TargetedConfig = Field(default_factory=TargetedConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)


class TranscriptionConfig(_Model):
    backend: Literal["faster-whisper", "whisperx"] = "faster-whisper"
    align_model: str = "auto"
    model: str = "auto"
    anchor_model: str = "base.en"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    beam_size: int = Field(5, ge=1, le=20)
    batch_size: int = Field(8, ge=1, le=64)
    vad_filter: bool = True
    initial_prompt: str = "auto"


class OpenSubtitlesConfig(_Model):
    enabled: bool = True
    api_key: str = ""
    username: str = ""
    password: str = ""
    user_agent: str = "video-beep-remover v0.1"
    exclude_machine_translated: bool = True


SubtitleSource = Literal["embedded", "sidecar", "opensubtitles"]
_DEFAULT_SOURCES: tuple[SubtitleSource, ...] = ("embedded", "sidecar", "opensubtitles")


class SubtitlesConfig(_Model):
    languages: list[str] = Field(default_factory=lambda: ["en"])
    sources: list[SubtitleSource] = Field(default_factory=lambda: list(_DEFAULT_SOURCES))
    prefer_hearing_impaired: bool = True
    max_candidates: int = Field(3, ge=1)
    opensubtitles: OpenSubtitlesConfig = Field(default_factory=OpenSubtitlesConfig)


class OutputConfig(_Model):
    path: str = "{stem}.clean{ext}"
    overwrite: bool = False
    when_clean: Literal["copy", "skip"] = "copy"
    audio_codec: str = "auto"
    audio_bitrate: str = "auto"
    other_audio_streams: Literal["auto", "censor", "copy", "drop"] = "auto"
    subtitle_streams: Literal["censor", "copy", "drop"] = "censor"
    subtitle_mask: Literal["first_letter", "asterisks", "remove"] = "first_letter"
    report: bool = True
    edl: bool = False
    review_srt: bool = False


class CacheConfig(_Model):
    dir: str = "auto"
    max_size_gb: float = Field(5, ge=0)
    transcripts: bool = True


class ToolsConfig(_Model):
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


class Config(_Model):
    config_version: Literal[1] = 1
    offline: bool = False
    lexicon: LexiconConfig = Field(default_factory=LexiconConfig)
    censor: CensorConfig = Field(default_factory=CensorConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
