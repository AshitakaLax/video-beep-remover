"""Mask listed words in subtitles for the cleaned output (DESIGN.md §6.11).

The words are found with the same matcher as the audio, in the text a viewer sees. Markup stays
exactly as it was: HTML-like tags (<i>, <font ...>), ASS override blocks ({\\an8}) and WebVTT tags
take no space when words are found, and ASS line breaks (\\N) separate words. SRT, WebVTT and ASS
files are edited in place, line by line, so styling, timing and layout survive untouched; other
formats go through pysubs2."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pysubs2

from video_beep_remover.detect.lexicon import Lexicon
from video_beep_remover.detect.matcher import Token, find_matches, runs
from video_beep_remover.detect.normalize import EDGE_PUNCTUATION, split_words
from video_beep_remover.errors import SubtitleError

Mask = Literal["first_letter", "asterisks", "remove"]
MASK_CHAR = "*"

_ZERO_WIDTH = re.compile(
    r"\{[^{}]*\}|</?[a-zA-Z][^<>]*>|<\d[\d:.]*>"
)  # override blocks, tags, VTT timestamps
_BREAK = re.compile(r"\\[Nnh]")  # ASS line breaks and hard spaces
_EDGE = EDGE_PUNCTUATION + "‘’‚‛′´`“”„″"  # stripped from both ends of a word, as normalize_token does
_TIMING = re.compile(r"^\s*(\d+:)?\d+:\d+[,.]\d+\s*-->")  # an SRT or WebVTT timing line


@dataclass(frozen=True)
class Censored:
    text: str
    masked: int  # how many listed words the text has, now masked (including ones masked already)


def _visible(text: str) -> tuple[str, list[int]]:
    """The text a viewer sees, and the position in `text` of each of its characters."""
    chars: list[str] = []
    positions: list[int] = []
    i = 0
    while i < len(text):
        tag = _ZERO_WIDTH.match(text, i)
        if tag:
            i = tag.end()
            continue
        if _BREAK.match(text, i):
            chars.append(" ")
            positions.append(i)
            i += 2
            continue
        chars.append(text[i])
        positions.append(i)
        i += 1
    return "".join(chars), positions


def _core(visible: str, start: int, end: int) -> tuple[int, int]:
    """The word without surrounding punctuation ("hell?" -> "hell")."""
    while start < end and visible[start] in _EDGE:
        start += 1
    while end > start and visible[end - 1] in _EDGE:
        end -= 1
    return start, end


def mask_text(text: str, lexicon: Lexicon, mask: Mask) -> Censored:
    """Mask the listed words in one cue's text. Hint words are never masked.

    - first_letter: "fucking" -> "f******" (already masked words such as "f***" stay as they are)
    - asterisks: "fucking" -> "*******"
    - remove: the word is deleted, with the space next to it"""
    visible, positions = _visible(text)
    words = split_words(visible)
    tokens = [Token.from_raw(word) for word, _, _ in words]
    replace: dict[int, str] = {}  # position in `text` -> replacement character ("" deletes it)
    masked = 0
    for match in find_matches(lexicon, tokens):
        for run in runs(match.targets):  # consecutive words of a phrase go together
            cores = [c for c in (_core(visible, words[i][1], words[i][2]) for i in run) if c[0] < c[1]]
            masked += len(cores)
            if not cores:
                continue
            if mask != "remove":
                for start, end in cores:
                    letters = [p for p in range(start, end) if visible[p].isalnum()]
                    for p in letters[1:] if mask == "first_letter" else letters:
                        replace[positions[p]] = MASK_CHAR
                continue
            start, end = cores[0][0], cores[-1][1]
            for p in range(start, end):
                if not _BREAK.match(text, positions[p]):  # a line break inside a phrase stays
                    replace[positions[p]] = ""
            before = visible[start - 1] if start > 0 else ""
            after = visible[end] if end < len(visible) else ""
            # Only a real space goes with the words, never an ASS line break (\N).
            if after == " " == text[positions[end]] and (not before or before.isspace()):
                replace[positions[end]] = ""  # "Fuck you" -> "you"
            elif before == " " == text[positions[start - 1]] and not after.isalnum():
                replace[positions[start - 1]] = ""  # "What the hell?" -> "What the?"
    if not replace:  # nothing listed, or only words that were masked already ("f***")
        return Censored(text, masked)
    return Censored("".join(replace.get(i, ch) for i, ch in enumerate(text)), masked)


def _censor_cue_blocks(text: str, lexicon: Lexicon, mask: Mask) -> Censored:
    """SRT and WebVTT: the lines after each timing line, up to the next blank line, are cue text."""
    out: list[str] = []
    cue: list[str] = []
    masked = 0

    def flush() -> None:
        nonlocal masked
        if cue:
            done = mask_text("".join(cue), lexicon, mask)
            out.append(done.text)
            masked += done.masked
            cue.clear()

    in_cue = False
    for line in text.splitlines(keepends=True):
        if not line.strip():
            flush()
            in_cue = False
            out.append(line)
        elif in_cue:
            cue.append(line)
        else:
            in_cue = bool(_TIMING.match(line))
            out.append(line)
    flush()
    return Censored("".join(out), masked)


def _censor_ass(text: str, lexicon: Lexicon, mask: Mask) -> Censored:
    """ASS and SSA: the last field of each Dialogue line in [Events] is its text."""
    out: list[str] = []
    masked = 0
    fields = 10
    in_events = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            in_events = stripped.lower() == "[events]"
        elif in_events and stripped.lower().startswith("format:"):
            fields = max(1, len(stripped.split(":", 1)[1].split(",")))
        elif in_events and stripped.lower().startswith("dialogue:"):
            head, _, rest = line.partition(":")
            parts = rest.split(",", fields - 1)
            if len(parts) == fields:
                body = parts[-1]
                ending = body[len(body.rstrip("\r\n")) :]
                done = mask_text(body[: len(body) - len(ending)], lexicon, mask)
                masked += done.masked
                line = f"{head}:{','.join([*parts[:-1], done.text])}{ending}"
        out.append(line)
    return Censored("".join(out), masked)


def subtitle_format(text: str) -> str:
    """ "srt", "vtt", "ass" or another format pysubs2 recognizes."""
    body = text.lstrip("﻿")
    if body.startswith("WEBVTT"):
        return "vtt"
    if re.search(r"^\s*\[events\]\s*$", body, re.IGNORECASE | re.MULTILINE) and re.search(
        r"^\s*dialogue:", body, re.IGNORECASE | re.MULTILINE
    ):
        return "ass"
    if any(_TIMING.match(line) for line in body.splitlines()[:50]):
        return "srt"
    try:
        from pysubs2.formats import autodetect_format

        return str(autodetect_format(body))
    except Exception as exc:  # untrusted input
        raise SubtitleError(f"unknown subtitle format ({exc})") from exc


def censor_subtitles(text: str, lexicon: Lexicon, mask: Mask, *, fps: float | None = None) -> Censored:
    """The whole subtitle file with listed words masked, in its own format."""
    fmt = subtitle_format(text)
    if fmt in ("srt", "vtt"):
        return _censor_cue_blocks(text, lexicon, mask)
    if fmt == "ass":
        return _censor_ass(text, lexicon, mask)
    try:
        subs = pysubs2.SSAFile.from_string(text, format_=fmt, fps=fps)
        masked = 0
        for event in subs:
            done = mask_text(event.text, lexicon, mask)
            event.text = done.text
            masked += done.masked
        return Censored(subs.to_string(fmt, fps=fps), masked)
    except Exception as exc:  # untrusted input
        raise SubtitleError(f"could not censor {fmt} subtitles: {exc}") from exc


def censored_copy_path(subtitle: Path, video: Path, output: Path, language: str | None) -> Path:
    """Where the censored copy of a subtitle file used for analysis goes: next to the output, named so
    players load it with the output ("Movie.en.srt" -> "Movie.clean.en.srt")."""
    stem = subtitle.stem
    if stem.lower().startswith(video.stem.lower()) and stem[len(video.stem) : len(video.stem) + 1] in "._-":
        rest = stem[len(video.stem) :]
    else:
        rest = f".{language}" if language else ""
    return output.with_name(f"{output.stem}{rest}{subtitle.suffix.lower()}")
