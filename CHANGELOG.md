# Changelog

All notable changes to video-beep-remover. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

The first release: the complete v1 of [the design](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md).

### Finding words

- A configurable word list with categories, wildcards, phrases, an allow list and masked spellings such as `f***` (`[lexicon]`).
- Speech recognition with faster-whisper, on an NVIDIA GPU or the CPU; the `[gpu]` extra installs the CUDA libraries it needs on Linux. On Windows it uses those of PyTorch's CUDA build, with nothing added to `PATH`, and they keep the cuDNN copy faster-whisper's library ships from breaking PyTorch's.
- Three strategies:
  - `hybrid` (the default) transcribes around subtitle lines with listed words, plus any speech no subtitle covers.
  - `targeted` transcribes around subtitle lines with listed words only.
  - `full` transcribes the whole soundtrack.
- Subtitles come from inside the file, from next to it, or from OpenSubtitles.com with your own API key. vbr's own review subtitles are never taken for a video's subtitles.
- OpenSubtitles is searched by the movie hash, then by an IMDb id from a Kodi-style `.nfo` file, then by title. An episode named without its show (`S01E02 - Pilot.mkv`) is searched under the show its `.nfo` file or its library folders name, never under its own title.
- Subtitles are used only if a sync check passes. The check finds offsets and frame-rate differences, and can re-time subtitles with the optional ffsubsync (`[sync]` extra).
- An optional `whisperx` backend (`[align]` extra) re-times words with wav2vec2 forced alignment.

### Muting

- Each word is muted with short fades inside its padding.
- Only the audio is re-encoded; video, chapters and attachments are copied untouched.
- Every muted span is checked for silence before the output is kept, and the output must keep every stream and the input's length.
- Other audio tracks get the same mutes when their dialogue matches the analysed track; tracks that don't match are dropped.
- Listed words are masked in text subtitle tracks, and in a copy of the subtitle file that guided the search, which keeps the file's line endings.
- An optional edge refinement (`censor.refine_edges`) moves each edge to the quietest 10 ms nearby.

### Output and workflow

- A JSON report for every file, with optional EDL and review-subtitle outputs.
- `vbr render` mutes exactly the spans in a hand-edited report, found next to each video or named with `--report`.
- Every command that takes videos takes files, folders, or both: `clean`, `scan`, `render` and `subs`. MKV and MP4 are tested; the output keeps the input's container.
- Folder batches keep the model loaded and render each file while the next one is analysed. Files vbr already censored, and its outputs and backups, are skipped. Any error in one file, even an unexpected one, fails only that file.
- OpenSubtitles answers that aren't JSON, and network errors while downloading, make the service unavailable for the file instead of failing it.
- `--backup` puts the cleaned file in the original's place and keeps the unmodified original as `Movie.orig.mp4`; `--in-place` does the same with no backup (`output.mode`). The original is only touched once the new file is verified.
- A transcript cache makes re-runs fast, for example after editing the word list.
- Each file's result goes to standard output, and messages and progress to standard error, both in UTF-8 when redirected.
- Temporary files no longer pile up: a file that fails leaves none behind, even on Windows, and the next run deletes those a killed run left, once they are 12 hours old.
- Other commands: `vbr doctor`, `vbr subs`, `vbr cache` and `vbr config init/show/check`.

### Context analysis (preview)

- `--context` (`[context]` extra) reads the dialogue in context with local models. By default it only reports: its verdicts go into the report and the review subtitles, and change nothing that is muted.
  - Each ambiguous listed word gets a verdict: profane, probably harmless ("the nine circles of hell"), or unsure.
  - Lines that look sexual are listed with their evidence: explicit wording, phrases, sound descriptions such as `[moaning]`, and innuendo the judge recognizes.
- Opt-in, experimental actions on the verdicts: `context.harmless = "keep"` leaves uses judged harmless unmuted, and `context.sexual = "mute"` mutes whole lines flagged as sexual. Only subtitle lines that were heard in the audio can show a use as harmless, which defeats notes and answers written into subtitles to sway the judge.
- A `sexual` word category of phrases of a sexual nature, off by default.
- `scripts/evaluate_context.py`, a labelled set of 71 lines and a set of crafted lines, to measure the context layer.
- `context.judge = "api"` asks a service online instead of a local judge (`[context.api]`): Google's Gemini API (`gemini-3.5-flash-lite`, which has a free tier), Jev's decision API, or any OpenAI-compatible chat API. Only the lines asked about are sent; `--offline` turns it off. An unanswered question mutes the word, and a service that fails three questions in a row isn't asked again in that run.
- The context and voice models run on the GPU only when PyTorch can use it. With a CPU-only build of PyTorch they run on the CPU, and vbr says so.
- `context.judge = "auto"` runs the judge only on a GPU with room for it, about 9 GB; on a smaller one the report says why there is none. The judge loads straight onto the GPU, not through system memory.

### Voice replacement (experimental)

- `--replace` (`[voice]` extra) says a milder word in the speaker's voice instead of muting, where a substitute from `[replace.substitutes]` fits: "damn" becomes "darn". F5-TTS says the word again inside its sentence, over the dialogue that Demucs separates from the music and effects.
- Every replaced word is checked: Whisper must hear the substitute and no listed word, and it must sound like the speaker. It is heard again in the rendered file, where no listed word may be heard. Anything else, including a model error on the word, is muted as before. The span stays muted in other audio tracks, the EDL and `vbr render`.
- Only one large model is in memory at a time: Whisper, the context models, Demucs or F5-TTS. Each is freed before the next loads, so replacement fits a 6 GB GPU. `models.keep_loaded = true` keeps them all loaded.
- `--offline` keeps the voice models off the network too: they load from the cache, or not at all.
- F5-TTS's model weights are licensed for non-commercial use only.

[Unreleased]: https://github.com/AshitakaLax/video-beep-remover/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AshitakaLax/video-beep-remover/releases/tag/v0.1.0
