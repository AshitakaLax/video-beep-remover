"""Build a small synthetic evaluation set: espeak-ng speech with exact word timings.

    python scripts/make_synthetic_set.py OUT_DIR

Each clip is a video (FLAC audio, so no encoder priming moves the timeline) with a ground-truth file
<stem>.truth.json and, for most clips, subtitles next to it. Every listed word is synthesized on its
own and joined to the rest of its sentence, so its start and end are known to the sample. The clips
cover what the design's evaluation set asks for, as far as synthetic speech can: verbatim, masked and
softened subtitles, lines the subtitles skip, subtitles out of sync, no subtitles at all, noise and a
music-like bed, and several voices and speeds. Synthetic speech is far cleaner than a film's
soundtrack: use this set to catch regressions, and a set of real clips to tune the defaults.

Needs espeak-ng and FFmpeg on PATH."""

import argparse
import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RATE = 22_050
WORD_GAP_S = 0.07  # between the synthesized parts of a sentence
LINE_GAP_S = 1.6

# A line is a list of parts; a part in [brackets] is a listed word, synthesized on its own.
LINES = [
    "We found the copper lantern near the old harbor.",
    "What the [hell] is going on over there?",
    "Every pilgrim needs a marble saddle for the journey.",
    "Get the [fuck] out of my orchard right now.",
    "The glacier violin sounds like distant thunder again.",
    "That was a [damn] fine cup of coffee, detective.",
    "Honestly the weather here is lovely in spring.",
    "This whole plan is [bullshit] and you know it.",
    "Hello, shell collectors, please assess the class schedule.",  # innocent near-misses
    "Freaking unbelievable, what the heck happened here?",  # hint words: never muted
    "Meadow falcons never sleep before the dawn arrives.",
    "You lying [bastard], I knew it all along.",
    "Cancel the parade, the orchestra is late again.",
    "Shut up, you stupid [ass], and listen to me.",
]
# Neutral dialogue between those lines: in a film, flagged lines are minutes apart, not seconds.
FILLER = [
    "The train to the coast leaves at seven in the morning.",
    "Did you remember to water the tomatoes before you left?",
    "My grandmother kept every letter she ever received.",
    "The museum is closed on Mondays, so let us go tomorrow.",
    "I think the map is upside down, look at the river.",
    "Nobody told me the meeting had been moved to Friday.",
    "The bakery on the corner sells the best bread in town.",
    "We should paint the fence before the summer rain starts.",
    "Can you hand me the blue folder on the second shelf?",
    "The lighthouse keeper waved at every passing boat.",
    "Her brother builds wooden clocks in his spare time.",
    "The soup needs a little more salt and some pepper.",
    "They walked along the beach until the sun went down.",
    "Please turn off the lights when you leave the office.",
    "The children built a castle out of cardboard boxes.",
    "I have never seen so many stars above the desert.",
    "The old bridge was repaired last autumn by the council.",
    "Your package arrived while you were at the dentist.",
    "We drove through three small villages on the way north.",
    "The captain checked the weather report twice before sailing.",
    "A quiet library is the best place to finish a novel.",
    "The garden looks wonderful now that the roses are blooming.",
    "Remember to bring a warm coat, the mountains get cold.",
    "The orchestra rehearsed the second movement all afternoon.",
    "I found your keys in the pocket of the green jacket.",
    "The market sells fresh fish every Tuesday and Saturday.",
    "Our neighbors adopted a very small and very loud puppy.",
    "The professor explained the theory with a simple drawing.",
]
FILLERS_BETWEEN = 3  # neutral lines after each line of LINES

# How subtitles show a line: masked and softened versions of the lines with listed words.
MASKED = {
    "hell": "h***",
    "fuck": "f***",
    "damn": "d***",
    "bullshit": "bulls***",
    "bastard": "b******",
    "ass": "a**",
}
SOFTENED = {
    "hell": "heck",
    "fuck": "frick",
    "damn": "darn",
    "bullshit": "nonsense",
    "bastard": "creep",
    "ass": "fool",
}


@dataclass
class ClipSpec:
    name: str
    voice: str = "en-us"
    speed: int = 150
    lines: list[int] = field(default_factory=lambda: list(range(len(LINES))))  # which of LINES
    subtitles: str = "verbatim"  # verbatim | masked | softened | none
    skipped: tuple[int, ...] = ()  # lines missing from the subtitles
    offset: float = 0.0  # subtitles this late
    noise_db: float | None = None  # pink noise at this SNR
    music: bool = False  # a bed of chords under the speech


CLIPS = [
    ClipSpec("verbatim-us"),
    ClipSpec("verbatim-gb-noise", voice="en-gb", speed=140, noise_db=15.0),
    ClipSpec("masked-music", voice="en-us+m3", subtitles="masked", music=True),
    ClipSpec("softened", voice="en-us+f3", subtitles="softened"),
    ClipSpec("unsubtitled-lines", voice="en-gb-x-rp", skipped=(3, 11)),
    ClipSpec("late-subtitles", voice="en-gb-scotland", speed=160, offset=1.7),
    ClipSpec("no-subtitles", voice="en-us+f2", subtitles="none"),
    ClipSpec("fast-noisy", voice="en-us+m1", speed=180, noise_db=10.0),
]


def synthesize(text: str, voice: str, speed: int, folder: Path) -> np.ndarray:
    path = folder / "part.wav"
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(path), text],
        check=True, capture_output=True,
    )  # fmt: skip
    with wave.open(str(path)) as audio:
        assert audio.getframerate() == RATE and audio.getsampwidth() == 2
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768


def trim(samples: np.ndarray, threshold: float = 10 ** (-40 / 20)) -> np.ndarray:
    """Without the silence espeak-ng puts around every utterance."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    return samples[loud[0] : loud[-1] + 1] if loud.size else samples[:0]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(spectrum.size), 1))
    noise = np.fft.irfft(spectrum, n)
    return (noise / np.sqrt(np.mean(noise**2))).astype(np.float32)


def chords(n: int) -> np.ndarray:
    t = np.arange(n) / RATE
    bed = np.zeros(n, np.float32)
    for start in range(0, int(t[-1]) + 1, 4):  # a new chord every 4 s
        root = [196.0, 220.0, 174.6, 261.6][start // 4 % 4]
        part = (t >= start) & (t < start + 4)
        for ratio in (1.0, 1.26, 1.5):
            bed[part] += np.sin(2 * np.pi * root * ratio * t[part]).astype(np.float32)
    return bed / 3


def srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def script(spec: ClipSpec, rng: np.random.Generator) -> list[tuple[str, int | None]]:
    """The clip's lines in order: (text, its number in LINES or None for a neutral line)."""
    order: list[tuple[str, int | None]] = []
    for number in spec.lines:
        order.append((LINES[number], number))
        order += [(FILLER[i], None) for i in rng.choice(len(FILLER), FILLERS_BETWEEN, replace=False)]
    return order


def build(spec: ClipSpec, out: Path, rng: np.random.Generator) -> None:
    pieces: list[np.ndarray] = [np.zeros(int(RATE * 1.0), np.float32)]
    position = 1.0
    truth: list[dict[str, object]] = []
    cues: list[tuple[float, float, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for line, number in script(spec, rng):
            start = position
            shown: list[str] = []
            for chunk in line.replace("[", "|[").replace("]", "]|").split("|"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                listed = chunk.startswith("[")
                word = chunk.strip("[]")
                audio = trim(synthesize(word, spec.voice, spec.speed, Path(tmp)))
                if listed:
                    truth.append(
                        {
                            "start": round(position, 3),
                            "end": round(position + len(audio) / RATE, 3),
                            "word": word,
                        }
                    )
                    shown.append(
                        {"masked": MASKED, "softened": SOFTENED}.get(spec.subtitles, {}).get(word, word)
                    )
                else:
                    shown.append(word)
                pieces += [audio, np.zeros(int(RATE * WORD_GAP_S), np.float32)]
                position += len(audio) / RATE + WORD_GAP_S
            if number is None or number not in spec.skipped:
                cues.append((start - 0.15 + spec.offset, position + 0.25 + spec.offset, " ".join(shown)))
            pieces.append(np.zeros(int(RATE * LINE_GAP_S), np.float32))
            position += LINE_GAP_S
    speech = np.concatenate(pieces)
    level = np.sqrt(np.mean(speech[np.abs(speech) > 1e-3] ** 2))
    mix = speech.copy()
    if spec.noise_db is not None:
        mix += pink_noise(mix.size, rng) * level / 10 ** (spec.noise_db / 20)
    if spec.music:
        mix += chords(mix.size) * level * 0.35
    mix /= max(1.0, float(np.max(np.abs(mix))) / 0.9)

    wav = out / f"{spec.name}.wav"
    with wave.open(str(wav), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes((mix * 32767).astype(np.int16).tobytes())
    video = out / f"{spec.name}.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x120:rate=24", "-i", str(wav), "-shortest", "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "flac", "-metadata:s:a:0", "language=eng",
         str(video)],
        check=True,
    )  # fmt: skip
    wav.unlink()
    (out / f"{spec.name}.truth.json").write_text(json.dumps({"words": truth}, indent=1) + "\n", "utf-8")
    if spec.subtitles != "none":
        text = "".join(
            f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1)
        )
        (out / f"{spec.name}.en.srt").write_text(text, "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    for spec in CLIPS:
        build(spec, args.out, rng)
        print(f"{spec.name}: done")


if __name__ == "__main__":
    main()
