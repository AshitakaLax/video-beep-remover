"""Read subtitle files and turn them into cleaned cues (DESIGN.md §6.5)."""

import codecs
import re
from pathlib import Path

import charset_normalizer
import pysubs2

from video_beep_remover.errors import SubtitleError
from video_beep_remover.models import Cue

MAX_FILE_BYTES = 20 * 1024 * 1024  # heavily typeset ASS files can be a few MB; nothing real is this big

_TAG = re.compile(r"</?[a-zA-Z][^<>]*>|<\d[\d:.]*>")  # leftover <font ...>, <c.yellow>, <00:01:02.000>
_OVERRIDE = re.compile(r"\{[^{}]*\}")  # leftover ASS override blocks such as {\an8}
_DESCRIPTION = re.compile(r"\[[^\[\]]*\]|\([^()]*\)")  # [door slams], (laughs), [JOHN]
_LABEL = re.compile(r"^([^\s:][^:]{0,29}):(?=\s|$)")  # JOHN: / MAN 2:
_DASH = re.compile(r"^[-‐‑‒–—]+\s*")  # dialogue dashes
_NOTES = ("♪", "♫", "♬")
_HASH_SONG = re.compile(r"^#\s.*\s#$|^#$")  # some subtitles mark songs with "#" instead of "♪"
_VTT_METADATA = ("NOTE", "STYLE", "REGION")
# The legacy Windows code page most subtitles in a language use when they are not UTF-8. Guessing the
# encoding of a short text is unreliable (French in cp1252 can look like Baltic cp1257), so the code
# page expected for the subtitle's language is tried first.
_CODE_PAGES = {
    **dict.fromkeys(("cs", "hu", "pl", "ro"), "cp1250"),
    **dict.fromkeys(("ru", "uk"), "cp1251"),
    "el": "cp1253", "tr": "cp1254", "he": "cp1255", "ar": "cp1256", "vi": "cp1258",
}  # fmt: skip


def decode_text(data: bytes, language: str | None = None) -> str:
    """Decode subtitle bytes: a BOM, then UTF-8, then the language's usual code page (cp1252 for
    Western European languages), then whatever charset-normalizer detects."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    if language:
        try:
            return data.decode(_CODE_PAGES.get(language.lower().split("-")[0], "cp1252"))
        except UnicodeDecodeError:
            pass
    best = charset_normalizer.from_bytes(data).best()
    return str(best) if best is not None else data.decode("latin-1")


def read_subtitle_file(path: Path, language: str | None = None) -> str:
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise SubtitleError(f"{path.name} is {size / 1e6:.0f} MB, too big for a subtitle file")
        return decode_text(path.read_bytes(), language)
    except OSError as exc:
        raise SubtitleError(f"could not read {path}: {exc.strerror or exc}") from exc


def _strip_label(line: str) -> str:
    """Remove a speaker label such as "JOHN:" (all capitals, so "Look: ..." survives)."""
    match = _LABEL.match(line)
    if match:
        label = match.group(1)
        if label.isupper() and len(label.split()) <= 4 and not any(ch in label for ch in '?!,"'):
            return line[match.end() :]
    return line


def clean_text(raw: str) -> tuple[str, bool]:
    """The spoken text of a cue as one line, and whether it is sung (DESIGN.md §6.5)."""
    lyrics = any(note in raw for note in _NOTES)
    text = _OVERRIDE.sub("", _TAG.sub("", raw))
    text = _DESCRIPTION.sub(" ", text)  # before splitting lines: descriptions can span two lines
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if _HASH_SONG.match(line):
            lyrics = True
            line = line.strip("#")
        line = _DASH.sub("", _strip_label(_DASH.sub("", line)))
        for note in _NOTES:
            line = line.replace(note, " ")
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return " ".join(lines), lyrics


def _without_vtt_metadata(text: str) -> str:
    """Drop WebVTT NOTE, STYLE and REGION blocks, which pysubs2 would read as cue text."""
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    kept = [block for block in blocks if not block.lstrip().startswith(_VTT_METADATA)]
    return "\n\n".join(kept) + "\n"


def parse_subtitles(text: str, *, fps: float | None = None) -> list[Cue]:
    """Parse SRT, ASS/SSA, WebVTT or MicroDVD (which needs `fps`) into cleaned cues, in time order.

    Comments, drawings and cues left empty by cleaning are dropped. The same line repeated in
    overlapping cues (e.g. on two ASS layers) becomes one cue.
    """
    text = text.lstrip("﻿")
    if text.startswith("WEBVTT"):
        text = _without_vtt_metadata(text)
    try:
        subs = pysubs2.SSAFile.from_string(text, fps=fps)
    except Exception as exc:  # untrusted input: any parser failure means "not usable"
        raise SubtitleError(f"not a subtitle file pysubs2 can read ({type(exc).__name__}: {exc})") from exc

    items: list[tuple[float, float, str, bool]] = []
    for event in subs:
        if event.is_comment or event.is_drawing or event.end <= event.start:
            continue
        spoken, lyrics = clean_text(event.plaintext)
        if spoken:
            items.append((event.start / 1000, event.end / 1000, spoken, lyrics))
    items.sort()

    merged: list[tuple[float, float, str, bool]] = []
    for start, end, spoken, lyrics in items:
        if merged and merged[-1][2] == spoken and start <= merged[-1][1] + 0.05:
            first = merged[-1]
            merged[-1] = (first[0], max(first[1], end), spoken, first[3] or lyrics)
        else:
            merged.append((start, end, spoken, lyrics))
    return [Cue(i + 1, start, end, spoken, lyrics) for i, (start, end, spoken, lyrics) in enumerate(merged)]
