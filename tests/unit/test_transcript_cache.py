import os
import time
from pathlib import Path

from video_beep_remover.asr.cache import EDGE_S, Transcript, TranscriptCache, settings_key
from video_beep_remover.models import Word

WORDS = tuple(Word(f"w{i}", i + 0.1, i + 0.4, 0.9) for i in range(10, 20))  # one word a second, 10-19 s


def store(tmp_path: Path, key: str = "k"):  # type: ignore[no-untyped-def]
    return TranscriptCache(tmp_path).store("abc-123", 1, "large-v3-turbo", key, 100.0)


def test_exact_windows_come_back_as_transcribed(tmp_path: Path) -> None:
    clip = Transcript(10.4, 19.6, WORDS, clean_start=True, clean_end=False)
    store(tmp_path).add_window(10.0, 20.0, clip)
    again = store(tmp_path)  # a new run reads the file
    assert again.window(10.0, 20.0) == Transcript(10.4, 19.6, WORDS, True, False)
    assert again.window(10.0, 20.5) is None  # neither the same window nor inside a reliable part


def test_windows_inside_a_longer_transcript_get_its_words(tmp_path: Path) -> None:
    cache = store(tmp_path)
    cache.add_window(10.0, 20.0, Transcript(10.0, 20.0, WORDS))
    inside = cache.window(12.0, 15.0)
    assert inside is not None and (inside.start, inside.end, inside.clean_start, inside.clean_end) == (
        12.0,
        15.0,
        True,
        True,
    )
    assert [w.text for w in inside.words] == ["w12", "w13", "w14"]
    assert cache.window(10.0 + EDGE_S - 0.1, 15.0) is None  # too close to an edge that may cut a word
    assert cache.window(10.0 + EDGE_S, 20.0 - EDGE_S) is not None


def test_a_whole_track_transcript_serves_every_window(tmp_path: Path) -> None:
    cache = store(tmp_path)
    assert cache.track() is None
    cache.add_track(WORDS, 99.95)  # the decoded audio ended a little before the container
    again = store(tmp_path)
    assert again.track() == list(WORDS)
    window = again.window(96.0, 100.0)
    assert window is not None and window.words == ()
    assert [w.text for w in (again.window(0.0, 11.0) or Transcript(0, 0, ())).words] == ["w10"]


def test_settings_and_broken_lines(tmp_path: Path) -> None:
    assert (
        settings_key(model="a", beam=5) == settings_key(beam=5, model="a") != settings_key(model="b", beam=5)
    )
    cache = store(tmp_path)
    cache.add_window(1.0, 5.0, Transcript(1.0, 5.0, ()))
    with cache.path.open("a", encoding="utf-8") as file:
        file.write('{"window": [30, 40], "clip": [30, 4')  # cut short by an interrupted run
    assert store(tmp_path).window(1.0, 5.0) is not None
    assert store(tmp_path, key="other").window(1.0, 5.0) is None  # other settings, other file


def test_speech_regions(tmp_path: Path) -> None:
    cache = TranscriptCache(tmp_path)
    assert cache.speech("abc-123", 1) is None
    cache.remember_speech("abc-123", 1, [(1.0, 2.5), (4.0, 9.25)])
    assert cache.speech("abc-123", 1) == [(1.0, 2.5), (4.0, 9.25)]
    assert cache.speech("abc-123", 2) is None


def test_least_recently_used_files_are_evicted(tmp_path: Path) -> None:
    cache = TranscriptCache(tmp_path)
    paths = []
    for number, name in enumerate(("old", "used", "new")):
        entry = cache.store(f"file-{number}", 1, "m", name, 60.0)
        entry.add_window(0.0, 30.0, Transcript(0.0, 30.0, WORDS))
        os.utime(entry.path, (time.time() - 100 + number, time.time() - 100 + number))
        paths.append(entry.path)
    cache.store("file-1", 1, "m", "used", 60.0).window(0.0, 30.0)  # reading it counts as a use
    size = paths[0].stat().st_size
    assert cache.usage() == (3, 3 * size)
    assert cache.evict(2 * size) == 1
    assert [p.exists() for p in paths] == [False, True, True]
    assert cache.evict(0) == 2 and cache.usage() == (0, 0)
    assert cache.clear() == 0


def test_cover_cuts_cached_pieces_to_the_window_and_lists_the_gaps(tmp_path: Path) -> None:
    cache = store(tmp_path)
    cache.add_window(8.0, 14.0, Transcript(8.0, 14.0, WORDS[:4]))  # w10-w13; reliable 8.3-13.7
    # Trimmed to speech at 16.8 s: the 0.3 s of silence before it counts as heard. Reliable 16.5-20.7.
    cache.add_window(16.5, 21.0, Transcript(16.8, 21.0, WORDS[6:], clean_start=True))
    pieces, gaps = cache.cover(10.0, 20.0)
    assert gaps == [(13.7, 16.5)]
    assert [(p.start, p.end, p.clean_start, p.clean_end) for p in pieces] == [
        (10.0, 14.0, True, False),  # cut inside the reliable part: clean; its own end is not
        (16.8, 20.0, True, True),
    ]
    assert [w.text for w in pieces[0].words] == ["w10", "w11", "w12", "w13"]
    assert [w.text for w in pieces[1].words] == ["w17", "w18", "w19"]
    assert cache.cover(30.0, 40.0) == ([], [(30.0, 40.0)])
    exact, none = cache.cover(8.0, 14.0)
    assert none == [] and exact[0].words == WORDS[:4]
    # A clip that starts where the window starts: its first 0.3 s is the window's own edge, which a
    # fresh transcription would not hear any better. So only the stretch after it is missing.
    assert cache.cover(8.0, 16.0)[1] == [(13.7, 16.0)]


def test_gaps_are_filled_with_overlap_and_a_minimum_length() -> None:
    from video_beep_remover.guided import OVERLAP_S, fill_gap
    from video_beep_remover.models import Window

    window = Window(10.0, 20.0)
    assert fill_gap((13.7, 16.8), window, 4.0) == (13.7 - OVERLAP_S, 16.8 + OVERLAP_S)
    assert fill_gap((19.9, 20.0), window, 4.0) == (16.0, 20.0)  # at least 4 s, inside the window
    assert fill_gap((10.0, 10.2), Window(10.0, 12.0), 4.0) == (10.0, 12.0)
