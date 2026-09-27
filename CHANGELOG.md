# Changelog

All notable changes to video-beep-remover. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

The first release: the complete v1 of [the design](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md).

### Finding words

- A configurable word list with categories, wildcards, phrases, an allow list and masked spellings such as `f***` (`[lexicon]`).
- Speech recognition with faster-whisper, on an NVIDIA GPU or the CPU; the `[gpu]` extra installs the CUDA libraries it needs.
- Three strategies:
  - `hybrid` (the default) transcribes around subtitle lines with listed words, plus any speech no subtitle covers.
  - `targeted` transcribes around subtitle lines with listed words only.
  - `full` transcribes the whole soundtrack.
- Subtitles come from inside the file, from next to it, or from OpenSubtitles.com with your own API key.
- Subtitles are used only if a sync check passes. The check finds offsets and frame-rate differences, and can re-time subtitles with the optional ffsubsync (`[sync]` extra).
- An optional `whisperx` backend (`[align]` extra) re-times words with wav2vec2 forced alignment.

### Muting

- Each word is muted with short fades inside its padding.
- Only the audio is re-encoded; video, chapters and attachments are copied untouched.
- Every muted span is checked for silence before the output is kept.
- Other audio tracks get the same mutes when their dialogue matches the analysed track; tracks that don't match are dropped.
- Listed words are masked in text subtitle tracks, and in a copy of the subtitle file that guided the search.
- An optional edge refinement (`censor.refine_edges`) moves each edge to the quietest 10 ms nearby.

### Output and workflow

- A JSON report for every file, with optional EDL and review-subtitle outputs.
- `vbr render` mutes exactly the spans in a hand-edited report.
- Folder batches keep the model loaded and render each file while the next one is analysed. Files vbr already censored are skipped.
- A transcript cache makes re-runs fast, for example after editing the word list.
- Other commands: `vbr doctor`, `vbr subs`, `vbr cache` and `vbr config init/show/check`.

### Context analysis (preview)

- `--context` (`[context]` extra) reads the dialogue in context with local models. It is report-only: its verdicts go into the report and the review subtitles, and never change what is muted.
  - Each ambiguous listed word gets a verdict: profane, probably harmless ("the nine circles of hell"), or unsure.
  - Lines that look sexual are listed with their evidence: explicit wording, phrases, sound descriptions such as `[moaning]`, and innuendo the judge recognizes.
- A `sexual` word category of phrases of a sexual nature, off by default.
- `scripts/evaluate_context.py` and a labelled set of 71 lines, to measure the context layer.

[Unreleased]: https://github.com/AshitakaLax/video-beep-remover/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AshitakaLax/video-beep-remover/releases/tag/v0.1.0
