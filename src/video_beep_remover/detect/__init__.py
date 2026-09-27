"""Finding listed words (DESIGN.md §6.7-6.10): the word list, matching, window planning, confirmation
and the muted spans."""

from video_beep_remover.detect.intervals import build_intervals
from video_beep_remover.detect.lexicon import Lexicon, compile_lexicon
from video_beep_remover.detect.matcher import Token, detect_in_words, find_matches

__all__ = ["Lexicon", "Token", "build_intervals", "compile_lexicon", "detect_in_words", "find_matches"]
