"""Configuration schema (DESIGN.md §4). The packaged defaults.toml fills in every field."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    mode: Literal["new", "backup", "in_place"] = "new"  # where the cleaned file goes (outputs.place)
    backup_path: str = "{stem}.orig{ext}"  # where "backup" keeps the original
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


_AMBIGUOUS = [
    "hell", "damned", "ass", "asses", "jackass*", "bitch*", "bastard*", "piss", "pissed", "jesus christ",
    "sleep with", "sleeping with", "slept with", "sleeps with", "sleep together", "slept together",
    "sleeping together", "hook up", "hooked up", "hooking up", "go down on", "went down on",
    "going down on", "get it on", "getting it on", "take off your clothes", "take your clothes off",
]  # fmt: skip
_TRIGGERS = [
    "bed", "naked", "nude", "undress*", "sexy", "seduc*", "virgin*", "lover*", "aroused", "horny",
    "spend the night", "come upstairs", "your place or mine", "take it off",
]  # fmt: skip


class ContextApiConfig(_Model):
    """A judge behind an API (context.judge = "api", DESIGN.md §17.10)."""

    provider: Literal["gemini", "jev", "openai"] = "gemini"  # "openai": any OpenAI-compatible chat API
    url: str = ""  # "": the provider's
    model: str = ""  # "": the provider's default
    fallback_models: list[str] = Field(default_factory=list)  # asked in order while those before are busy
    api_key: str = ""
    reasoning_effort: str = "auto"  # "auto": Gemini thinks little ("none" on 2.5); "": never sent
    timeout_s: float = Field(30, gt=0)


class ContextConfig(_Model):
    enabled: bool = False
    harmless: Literal["report", "keep"] = "report"  # "keep": leave uses judged harmless unmuted
    sexual: Literal["report", "mute"] = "report"  # "mute": mute the lines flagged as sexual
    classifier: str = "unitary/unbiased-toxic-roberta"
    judge: str = "auto"  # "auto": the default judge on a CUDA GPU, none on a CPU; "api"; "": none
    api: ContextApiConfig = Field(default_factory=ContextApiConfig)
    ambiguous: list[str] = Field(default_factory=lambda: list(_AMBIGUOUS))
    triggers: list[str] = Field(default_factory=lambda: list(_TRIGGERS))
    min_sexual_score: float = Field(0.5, ge=0, le=1)
    clean_below: float = Field(0.3, ge=0, le=1)
    profane_above: float = Field(0.5, ge=0, le=1)
    min_heard: float = Field(0.7, ge=0, le=1)  # share of a subtitle line's words heard, to trust it

    @model_validator(mode="after")
    def _api_complete(self) -> "ContextConfig":
        api = self.api
        if self.judge == "api" and not api.api_key:
            raise ValueError('judge = "api" needs context.api.api_key (e.g. "${VBR_JUDGE_API_KEY}")')
        if self.judge == "api" and api.provider == "openai" and not (api.url and api.model):
            raise ValueError('context.api.provider = "openai" needs context.api.url and context.api.model')
        return self


_SUBSTITUTES = {
    "*fuck*": ["freaking", "frick", "fricking", "fudge"],
    "*shit*": ["shoot", "shucks", "baloney", "crummy"],
    "bitch*": ["witch"],
    "son of a bitch": ["son of a gun"],
    "sonofabitch": ["son of a gun"],
    "*asshole*": ["jerk"],
    "bastard*": ["rascal"],
    "dickhead*": ["dummy"],
    "damn": ["darn"],
    "damned": ["darned"],
    "dammit": ["darn it"],
    "damnit": ["darn it"],
    "damn it": ["darn it"],
    "hell": ["heck"],
    "crap": ["crud"],
    "crappy": ["crummy"],
    "ass": ["butt"],
    "asses": ["butts"],
    "dumbass*": ["dummy"],
    "jackass*": ["jerk"],
    "smartass*": ["smarty-pants"],
    "pissed": ["ticked"],
    "goddamn*": ["gosh darn"],
    "god damn*": ["gosh darn"],
    "god dammit": ["gosh darn it"],
    "jesus christ": ["jeez"],
    "oh my [god]": ["gosh"],
    "for christ's sake": ["for crying out loud"],
}


class ReplaceConfig(_Model):
    enabled: bool = False
    model: str = "F5TTS_v1_Base"  # the voice editing model (F5-TTS)
    separation: str = "htdemucs"  # the model that splits the dialogue from music and effects
    steps: int = Field(32, ge=4, le=64)  # the voice model's sampling steps
    voice_margin: float = Field(0.15, ge=0, le=2)  # how much less like the speaker a new word may sound
    substitutes: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in _SUBSTITUTES.items()}
    )


class ModelsConfig(_Model):
    keep_loaded: bool = False  # true: keep every model in memory; false: one large model at a time


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
    context: ContextConfig = Field(default_factory=ContextConfig)
    replace: ReplaceConfig = Field(default_factory=ReplaceConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
