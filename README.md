# video-beep-remover

`vbr` is a command-line tool that mutes a configurable list of curse words in video files.

- **Speech recognition:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) finds each word and its timing.
- **Subtitles:** subtitles in the file, next to it, or from OpenSubtitles.com show where listed words are likely, so only the audio around those lines needs transcribing.
- **Rendering:** FFmpeg mutes each word with a short fade and re-encodes only the audio track. Video, other streams and chapters are copied untouched.
- **Checking:** every muted span in the output is verified to be silent before the file is kept.

See [docs/DESIGN.md](docs/DESIGN.md) for the design and [docs/vbr.example.toml](docs/vbr.example.toml) for every configuration option.

## Status

Milestones M0 to M4 of the [delivery plan](docs/DESIGN.md#13-delivery-plan) are implemented:

- the full-transcription pipeline, configuration, reports and EDL output
- subtitle-guided search with embedded and sidecar subtitles: the `hybrid` (default) and `targeted` strategies
- online subtitles from OpenSubtitles.com, with your own free API key, and an optional re-sync with ffsubsync
- the rest of v1: listed words masked in the output's subtitles, review subtitles, `vbr render` from an edited report, checks on other audio tracks, folder batches, and a transcript cache that makes re-runs fast

A file without usable subtitles falls back to transcribing the whole soundtrack, as §7 of the design describes, or fails with `--no-fallback`.

The defaults have been checked on a synthetic evaluation set only (see [Evaluate](#evaluate)); tuning them on real film clips is still to do, along with M5 (WhisperX, packaging and release).

## Install

You need Python 3.11+ and FFmpeg 5.1+ (`ffmpeg` and `ffprobe` on your `PATH`).

```console
$ pipx install git+https://github.com/AshitakaLax/video-beep-remover
$ vbr doctor          # checks FFmpeg, the mute filter, Whisper and credentials
```

To also re-time subtitles that are badly out of sync, install the optional [ffsubsync](https://github.com/smacke/ffsubsync) extra: `pipx install "video-beep-remover[sync] @ git+https://github.com/AshitakaLax/video-beep-remover"`.

Whisper runs on an NVIDIA GPU when CUDA 12 and cuDNN 9 are available, and on the CPU otherwise. The model downloads on first use.

## Use

```console
$ vbr clean "The Movie (2019).mkv"
  → The Movie (2019).clean.mkv        the copy with listed words muted
  → The Movie (2019).clean.vbr.json   what was found and muted, and when

$ vbr scan "The Movie (2019).mkv" --edl   # detect only; writes a report and a mute list for Kodi/MPlayer
$ vbr clean ~/Videos -r -o ~/Clean        # a whole folder, recursively
$ vbr subs "The Movie (2019).mkv"         # which subtitles would guide the search, and are they in sync?
$ vbr render "The Movie (2019).mkv" --report edited.vbr.json   # mute exactly the spans in a report
```

What the cleaned file contains:

- **Audio.** The dialogue track has each listed word muted. Another track in the same language, such as a stereo downmix, gets the same mutes if its audio matches the dialogue track's around the muted words; tracks in other languages, commentaries and mismatched tracks are dropped, since they would still contain the words (`output.other_audio_streams`).
- **Subtitles.** Listed words are masked in every text subtitle track (`f***`, or `****` or removed with `output.subtitle_mask`), keeping their styling and timing. Image-based subtitles (PGS, VobSub) are copied as they are, with a warning. When a subtitle file next to the video guided the search, a masked copy is written next to the output, e.g. `The Movie (2019).clean.en.srt`.
- **Everything else** (video, chapters, attachments, metadata) is copied untouched, and the file is tagged `VBR_CENSORED`, so later batch runs skip it.

Useful options:

- `--strategy hybrid|targeted|full` picks how words are found (see below), and `--no-fallback` fails instead of transcribing everything.
- `--subtitles FILE` uses that subtitle file instead of searching for one.
- `--categories strong,religious` enables exactly those word categories.
- `--model large-v3-turbo` picks a Whisper model, and `--device cpu` forces the CPU.
- `--audio-stream N` picks the dialogue track by its ffprobe index.
- `--review-srt` also writes `.review.srt`: one subtitle per muted span naming the word, to spot-check the result in a player.
- `--overwrite` or `--skip-existing` decide what happens when outputs already exist.

Run `vbr clean --help` for everything.

**Folders.** Every video in a folder is processed in turn (`-r` for subfolders). vbr's own outputs found there are skipped: files named like another input's output (`Movie.clean.mkv`) and files tagged `VBR_CENSORED`. The Whisper model stays loaded, and each file is written in the background while the next one is analysed. A file that fails doesn't stop the batch; the exit code is then 4.

**Editing the result.** Every run writes a JSON report whose `intervals` list the muted spans. Edit them (add, remove or move spans) and run `vbr render VIDEO --report REPORT` to mute exactly those, without detecting anything. A report made for a different file is refused unless you add `--force`.

## How words are found

1. **Subtitles.** vbr looks for text subtitles inside the file, then next to it (`Movie.en.srt`, `Movie.en.sdh.srt`, or a `Subs/` folder), and only then on OpenSubtitles.com (see below). Forced, wrong-language and machine-translated tracks are skipped, and SDH tracks are preferred because they are closer to verbatim.
2. **Sync check.** It transcribes a few seconds around six lines with a small model. The subtitles are used only if those lines are heard where the subtitles place them and nearly word for word. Subtitles not made for this exact file are searched within ±20 s, and a known frame-rate difference (e.g. a 25 fps release) is corrected up front. If the check fails on timing and ffsubsync is installed, the subtitles are re-timed with it and checked again.
3. **Windows.** Only the audio around lines that contain a listed word, a masked word such as `f***`, or a hint word such as "freaking" is transcribed. `hybrid`, the default, also transcribes any speech that no subtitle covers, such as songs and background voices. `targeted` skips that step.
4. **Confirmation.** A listed word the subtitles show but the audio doesn't confirm is looked for again in a wider window. If it still isn't heard, its time is estimated from its position in the line (`on_unconfirmed` also offers `cue` and `skip`).

vbr transcribes the whole soundtrack instead when:

- no subtitles pass the check
- the windows would cover more than 35 % of the runtime
- the audio confirms fewer than half of the flagged lines

With `--no-fallback`, the first and last cases fail the file instead. Many windows are still cheaper than everything, so they are transcribed. The JSON report records which subtitles were tried and why a fallback happened.

## Online subtitles

OpenSubtitles.com is searched only when no local subtitles are usable. It needs your own free API key: create an account on [opensubtitles.com](https://www.opensubtitles.com), register an API consumer, and put the key in the environment:

```console
$ export OPENSUBTITLES_API_KEY=...
$ export OPENSUBTITLES_USERNAME=... OPENSUBTITLES_PASSWORD=...   # optional: a larger daily download quota
$ vbr doctor                                                     # checks that the key is accepted
```

- **What is sent.** The file's OpenSubtitles hash (computed from 128 KiB of the file) and its size, then, if nothing matches the hash, the title and year or the season and episode from the file name. `vbr subs` shows exactly what it sends. `--offline` sends nothing.
- **Quota.** Searches are free; each download counts against a daily quota (5 without logging in). Only the subtitles that are actually tried are downloaded, and when the quota runs out the run falls back to transcribing everything. The report says when the quota renews.
- **Cache.** Downloaded subtitles are kept in your cache folder, so a file never costs quota twice, and they are reused even offline or without a key. `vbr cache clear --subtitles` deletes them.

## Re-running is fast

vbr keeps what Whisper heard in its cache, per file, model and settings. Running the same file again, for example after adding words to the list, needs little or no speech recognition: windows it has heard before come from the cache, only new stretches are transcribed, and the model isn't even loaded when nothing is new. On a synthetic test film, a second run took 0.1 s instead of 12 s, and a run with two words added took 6.6 s instead of 14.6 s.

The transcripts are deleted least recently used first beyond `cache.max_size_gb` (5 GB). Set `cache.transcripts = false` to keep none. `vbr cache info` shows what the cache holds, and `vbr cache clear` empties it (`--subtitles` or `--transcripts` for one part).

## Configure

```console
$ vbr config init     # writes every option, with comments, to your per-user config file
$ vbr config check    # validates it and compiles the word list
$ vbr config show     # prints the effective configuration (secrets redacted)
```

The word list supports whole words, `*` wildcards, phrases, `[bracketed]` targets inside a phrase (for example `oh my [god]`), regular expressions and an allowlist. Words masked as `f***` are detected too. A config file that defines its own categories replaces the built-in list.

## Develop

```console
$ pip install -e ".[dev]"
$ pytest                           # FFmpeg tests skip themselves when FFmpeg is missing
$ VBR_RUN_ASR_TESTS=1 pytest       # also runs real Whisper on synthesized speech (needs espeak-ng)
$ ruff check src tests && ruff format --check src tests && mypy
```

## License

MIT
