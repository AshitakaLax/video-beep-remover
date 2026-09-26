# video-beep-remover

A planned command-line tool that mutes a configurable list of curse words in video files.

- **Speech recognition:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) gives word-accurate timing.
- **Rendering:** FFmpeg rewrites only the audio track; video is copied untouched.
- **Speed:** subtitles (embedded, next to the file, or from OpenSubtitles.com) mark where the listed words occur. Only those few seconds of the soundtrack are transcribed, instead of the whole film.

**Status:** design phase. See:

- [docs/DESIGN.md](docs/DESIGN.md): the design (CLI, configuration, pipeline, performance, test plan)
- [docs/vbr.example.toml](docs/vbr.example.toml): the configuration file format, including the word list

```console
$ vbr config init                      # write a starter config
$ vbr clean "The Movie (2019).mkv"     # → The Movie (2019).clean.mkv + a JSON report
```

## License

MIT
