# video-beep-remover

`vbr` is a command-line tool that mutes a configurable list of curse words in video files.

- **Speech recognition:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) finds each word and its timing.
- **Rendering:** FFmpeg mutes each word with a short fade and re-encodes only the audio track. Video, other streams and chapters are copied untouched.
- **Checking:** every muted span in the output is verified to be silent before the file is kept.

See [docs/DESIGN.md](docs/DESIGN.md) for the design and [docs/vbr.example.toml](docs/vbr.example.toml) for every configuration option.

## Status

Milestones M0 and M1 of the [delivery plan](docs/DESIGN.md#13-delivery-plan) are implemented: the full-transcription pipeline, configuration, reports and EDL output. Subtitle-guided search (the `hybrid` and `targeted` strategies) comes in M2. Until then those strategies fall back to transcribing the whole soundtrack, as §7 of the design describes, or fail with `--no-fallback`.

Subtitle streams are dropped from the output for now, because subtitle censoring comes later (M4). Set `output.subtitle_streams = "copy"` to keep them uncensored.

## Install

You need Python 3.11+ and FFmpeg 5.1+ (`ffmpeg` and `ffprobe` on your `PATH`).

```console
$ pipx install git+https://github.com/AshitakaLax/video-beep-remover
$ vbr doctor          # checks FFmpeg, the mute filter, Whisper and credentials
```

Whisper runs on an NVIDIA GPU when CUDA 12 and cuDNN 9 are available, and on the CPU otherwise. The model downloads on first use.

## Use

```console
$ vbr clean "The Movie (2019).mkv"
  → The Movie (2019).clean.mkv        the copy with listed words muted
  → The Movie (2019).clean.vbr.json   what was found and muted, and when

$ vbr scan "The Movie (2019).mkv" --edl   # detect only; writes a report and a mute list for Kodi/MPlayer
$ vbr clean ~/Videos -r -o ~/Clean        # a whole folder, recursively
```

Useful options:

- `--categories strong,religious` enables exactly those word categories.
- `--model large-v3-turbo` picks a Whisper model, and `--device cpu` forces the CPU.
- `--audio-stream N` picks the dialogue track by its ffprobe index.
- `--overwrite` or `--skip-existing` decide what happens when outputs already exist.

Run `vbr clean --help` for everything.

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
