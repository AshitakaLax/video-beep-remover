import numpy as np

from video_beep_remover.asr.base import Clip
from video_beep_remover.asr.vad import snap_to_speech, trim_to_speech
from video_beep_remover.media.audio import SAMPLE_RATE
from video_beep_remover.models import Word

SPEECH = [(10.0, 12.5), (14.0, 16.0)]  # two lines with a 1.5 s pause between them


def test_a_word_placed_in_a_pause_starts_where_speech_resumes() -> None:
    words = [
        Word("line", 11.9, 12.4),
        Word("the", 12.6, 14.3),  # heard at 14.0, placed at the end of the previous line
        Word("glacier", 14.3, 14.8),
        Word("cough", 13.0, 13.3),  # entirely in the pause: left alone
        Word("close", 13.8, 14.2),  # less than MIN_SHIFT_S early: left alone
    ]
    assert snap_to_speech(words, SPEECH) == [
        Word("line", 11.9, 12.4),
        Word("the", 14.0, 14.3),
        Word("glacier", 14.3, 14.8),
        Word("cough", 13.0, 13.3),
        Word("close", 13.8, 14.2),
    ]
    assert snap_to_speech(words, []) == words


def test_trimming_reports_the_speech_in_media_time() -> None:
    clip = Clip(100.0, np.zeros(10 * SAMPLE_RATE, dtype=np.float32))
    trimmed = trim_to_speech(clip, lambda audio, on_progress=None: [(2.0, 4.0), (6.0, 7.5)])
    assert trimmed.speech == [(102.0, 104.0), (106.0, 107.5)]
    assert (trimmed.clip.start, round(trimmed.clip.duration, 3)) == (101.9, 5.7)
    assert trimmed.clean_start and trimmed.clean_end
    untouched = trim_to_speech(clip, lambda audio, on_progress=None: [])
    assert untouched.clip is clip and untouched.speech == [] and not untouched.clean_start
