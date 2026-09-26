"""Language codes as they appear in the config, container tags and subtitle file names."""

# ISO 639-1 codes (as used in the config) and the ISO 639-2 tags containers use.
_ISO639: dict[str, tuple[str, ...]] = {
    "ar": ("ara",), "cs": ("ces", "cze"), "da": ("dan",), "de": ("deu", "ger"), "el": ("ell", "gre"),
    "en": ("eng",), "es": ("spa",), "fi": ("fin",), "fr": ("fra", "fre"), "he": ("heb",),
    "hi": ("hin",), "hu": ("hun",), "id": ("ind",), "it": ("ita",), "ja": ("jpn",), "ko": ("kor",),
    "nl": ("nld", "dut"), "no": ("nor", "nob", "nno"), "pl": ("pol",), "pt": ("por",),
    "ro": ("ron", "rum"), "ru": ("rus",), "sv": ("swe",), "th": ("tha",), "tr": ("tur",),
    "uk": ("ukr",), "vi": ("vie",), "zh": ("zho", "chi"),
}  # fmt: skip
# Language names used in subtitle file names ("Movie.English.srt", "Subs/2_English.srt").
_NAMES: dict[str, str] = {
    "arabic": "ar", "czech": "cs", "danish": "da", "german": "de", "deutsch": "de", "greek": "el",
    "english": "en", "spanish": "es", "espanol": "es", "finnish": "fi", "french": "fr",
    "francais": "fr", "hebrew": "he", "hindi": "hi", "hungarian": "hu", "indonesian": "id",
    "italian": "it", "japanese": "ja", "korean": "ko", "dutch": "nl", "norwegian": "no",
    "polish": "pl", "portuguese": "pt", "brazilian": "pt", "romanian": "ro", "russian": "ru",
    "swedish": "sv", "thai": "th", "turkish": "tr", "ukrainian": "uk", "vietnamese": "vi",
    "chinese": "zh",
}  # fmt: skip


def lang_matches(tag: str | None, code: str) -> bool:
    """Compare a container language tag ("eng") with a config language code ("en")."""
    if not tag:
        return False
    tag_base = tag.lower().split("-")[0]
    code_base = code.lower().split("-")[0]
    if tag_base == code_base:
        return True
    return tag_base in _ISO639.get(code_base, ()) or code_base in _ISO639.get(tag_base, ())


def language_code(token: str) -> str | None:
    """The ISO 639-1 code for a language token ("en", "eng" or "English"), or None."""
    token = token.lower()
    if token in _ISO639:
        return token
    for code, tags in _ISO639.items():
        if token in tags:
            return code
    return _NAMES.get(token)
