# video-beep-remover

`vbr` is a command-line tool that mutes a configurable list of curse words in video files.

- **Speech recognition:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) finds each word and its timing.
- **Subtitles:** subtitles in the file, next to it, or from OpenSubtitles.com show where listed words are likely, so only the audio around those lines needs transcribing.
- **Rendering:** FFmpeg mutes each word with a short fade and re-encodes only the audio track. Video, other streams and chapters are copied untouched.
- **Checking:** every muted span in the output is verified to be silent before the file is kept.

See [docs/DESIGN.md](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md) for the design and [docs/vbr.example.toml](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/vbr.example.toml) for every configuration option.

## Status

All milestones of the [delivery plan](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md#13-delivery-plan), M0 to M5, are implemented:

- the full-transcription pipeline, configuration, reports and EDL output
- subtitle-guided search with embedded and sidecar subtitles: the `hybrid` (default) and `targeted` strategies
- online subtitles from OpenSubtitles.com, with your own free API key, and an optional re-sync with ffsubsync
- the rest of v1: listed words masked in the output's subtitles, review subtitles, `vbr render` from an edited report, checks on other audio tracks, folder batches, and a transcript cache that makes re-runs fast
- polish: an optional WhisperX backend and edge refinement for tighter word edges, GPU libraries installable with pip, and a release pipeline to PyPI

A file without usable subtitles falls back to transcribing the whole soundtrack, as §7 of the design describes, or fails with `--no-fallback`.

The defaults have been checked on a synthetic evaluation set only (see [Evaluate](#evaluate)); tuning them on real film clips is still to do. Changes are listed in the [changelog](https://github.com/AshitakaLax/video-beep-remover/blob/main/CHANGELOG.md).

A first version of context analysis (M6 and M7, [§17 of the design](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md#17-context-analysis-design-iteration)) is in: see [Context analysis](#context-analysis-preview). By default it only reports; acting on its verdicts is opt-in and experimental. So is [voice replacement](#voice-replacement-experimental) (M8), which says a milder word in the speaker's voice instead of muting.

## Install

You need Python 3.11+ and FFmpeg 5.1+ (`ffmpeg` and `ffprobe` on your `PATH`).

```console
$ pipx install video-beep-remover
$ vbr doctor          # checks FFmpeg, the mute filter, Whisper and credentials
```

The development version installs from GitHub: `pipx install git+https://github.com/AshitakaLax/video-beep-remover`.

Whisper runs on an NVIDIA GPU when CUDA 12 and cuDNN 9 are available, and on the CPU otherwise. The model downloads on first use. Optional extras add more, e.g. `pipx install "video-beep-remover[gpu,sync]"`:

| Extra | Adds |
|---|---|
| `gpu` | The cuBLAS and cuDNN libraries that Whisper needs on NVIDIA GPUs, on Linux. vbr loads them itself, so there is no `LD_LIBRARY_PATH` to set. The NVIDIA driver is still needed. |
| `align` | [WhisperX](https://github.com/m-bain/whisperX) forced alignment, for tighter word edges (see [Word edges](#word-edges)). It brings PyTorch, a large download. |
| `sync` | [ffsubsync](https://github.com/smacke/ffsubsync), to re-time subtitles that are badly out of sync. |
| `context` | PyTorch and transformers, for [context analysis](#context-analysis-preview). On Linux without a GPU, install the CPU build of PyTorch first to save gigabytes. |
| `voice` | F5-TTS, Demucs and SpeechBrain, for [voice replacement](#voice-replacement-experimental). A large install; F5-TTS's model weights are licensed for non-commercial use only. |

The `context` and `voice` extras run on PyTorch, which uses an NVIDIA GPU only in its CUDA build. On Windows, PyPI has the CPU build only, so install PyTorch's CUDA build from [pytorch.org](https://pytorch.org) first. When PyTorch cannot use the GPU that Whisper uses, vbr runs these models on the CPU and says so.

## Use

```console
$ vbr clean "The Movie (2019).mkv"
  → The Movie (2019).clean.mkv        the copy with listed words muted
  → The Movie (2019).clean.vbr.json   what was found and muted, and when

$ vbr clean ~/Videos -r -o ~/Clean        # a whole folder, recursively, into another folder
$ vbr clean ~/Videos -r --backup          # in place: Movie.mp4 is cleaned, Movie.orig.mp4 is the original
$ vbr scan "The Movie (2019).mp4" --edl   # detect only; writes a report and a mute list for Kodi/MPlayer
$ vbr subs ~/Videos                       # which subtitles would guide the search, and are they in sync?
$ vbr render "The Movie (2019).mkv"       # mute exactly the spans in its (edited) report
```

Every command that takes videos takes files, folders, or both (`-r` searches subfolders). MKV and MP4 are the tested containers; the others FFmpeg writes (MOV, WebM, TS, AVI…) are processed the same way. The cleaned file keeps its container: `.mp4` in, `.mp4` out.

What the cleaned file contains:

- **Audio.** The dialogue track has each listed word muted. Another track in the same language, such as a stereo downmix, gets the same mutes if its audio matches the dialogue track's around the muted words; tracks in other languages, commentaries and mismatched tracks are dropped, since they would still contain the words (`output.other_audio_streams`).
- **Subtitles.** Listed words are masked in every text subtitle track (`f***`, or `****` or removed with `output.subtitle_mask`), keeping their styling and timing; in an MP4 they stay `mov_text` tracks. Image-based subtitles (PGS, VobSub) are copied as they are, with a warning. When a subtitle file next to the video guided the search, a masked copy is written next to the output, e.g. `The Movie (2019).clean.en.srt`.
- **Everything else** (video, chapters, attachments, metadata) is copied untouched, and the file is tagged `VBR_CENSORED`, so later batch runs skip it.

Useful options:

- `--strategy hybrid|targeted|full` picks how words are found (see below), and `--no-fallback` fails instead of transcribing everything.
- `--subtitles FILE` uses that subtitle file instead of searching for one.
- `--categories strong,religious` enables exactly those word categories.
- `--model large-v3-turbo` picks a Whisper model, and `--device cpu` forces the CPU.
- `--audio-stream N` picks the dialogue track by its ffprobe index.
- `--review-srt` also writes `.review.srt`: one subtitle per muted span naming the word, to spot-check the result in a player. vbr never takes it for a video's subtitles.
- `--context` adds context analysis to the report ([preview](#context-analysis-preview)), and `--replace` says a milder word instead of muting ([experimental](#voice-replacement-experimental)).
- `--backup` or `--in-place` put the cleaned file in the original's place (see below).
- `--overwrite` or `--skip-existing` decide what happens when outputs already exist.

Run `vbr clean --help` for everything. Each file's result goes to standard output, and messages and progress to standard error, so `vbr scan ~/Videos -r > results.txt` keeps the results alone.

**Folders.** Every video in a folder is processed in turn (`-r` for subfolders). vbr's own files found there are skipped: files named like another input's output (`Movie.clean.mkv`) or backup (`Movie.orig.mkv`), and files tagged `VBR_CENSORED`. The Whisper model stays loaded, and each file is written in the background while the next one is analysed (not one with replaced words, which are heard again once written). A file that fails doesn't stop the batch; the exit code is then 4. Two inputs that would be written to the same output (say `Season 1/Episode 01.mkv` and `Season 2/Episode 01.mkv` with `-o ~/Clean`) are caught before anything is rendered: the later one fails.

**Replacing the originals.** By default the original is never changed: the cleaned copy is a new file next to it (`output.path`, or `-o`). Two flags put the cleaned file in the original's place instead, so a media library plays it without any change (`output.mode` sets either one for good):

- `--backup` keeps the unmodified original as `Movie.orig.mp4` (`output.backup_path`). A file whose backup already exists was cleaned before and is skipped, so a backup is never overwritten.
- `--in-place` keeps no backup.

Either way, the new file is written and verified before the original is touched, so a failed run leaves it as it was. A subtitle file next to the video that guided the search is masked in place too (with `--backup`, kept as `Movie.orig.en.srt`). To clean a file again from scratch, for example with a new word list, move its backup back over it first.

**Editing the result.** Every run writes a JSON report whose `intervals` list the muted spans. Edit them (add, remove or move spans) and run `vbr render VIDEO` to mute exactly those, without detecting anything: it reads the report that `vbr scan` or `vbr clean` wrote next to the video, or the one `--report` names. A report made for a different file is refused unless you add `--force`.

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

## Word edges

Whisper's word times are approximate, so each muted span is padded: 120 ms before the word and 200 ms after it (`censor.pad_before_ms`, `censor.pad_after_ms`). On the synthetic evaluation set, Whisper placed word ends up to 230 ms early, and starts early too. Two optional settings work on the edges:

- **WhisperX** (`transcription.backend = "whisperx"`, with the `align` extra). After Whisper transcribes, a wav2vec2 model aligns each word to the audio. On the evaluation set, word ends then came within 50 ms of the truth, where Whisper's came up to 230 ms early. Aligned starts came up to 200 ms late, though, so vbr keeps the earlier of the two starts and the later of the two ends: alignment only ever widens a word. With the default padding, that mutes a little more around each word. `pad_after_ms = 120` still fully muted every word the default did. On a CPU, `hybrid` took 18 % longer, plus a few seconds to load PyTorch. English, French, German, Spanish, Italian and 36 more languages have an alignment model; for others, name one with `transcription.align_model`.
- **Edge refinement** (`censor.refine_edges = true`). Each edge of a muted span moves outward, by up to 80 ms, to the quietest 10 ms nearby, so a fade doesn't cut a syllable in half. On the evaluation set it made no difference with the default padding. With `pad_after_ms = 120`, it raised the share of fully muted words from 69 % to 88 %.

## Context analysis (preview)

`--context` asks local models to read the dialogue around each listed word, and the whole script, and adds their verdicts to the report and the review subtitles. **By default it never changes what is muted**: the verdicts are there for you to check. Acting on them is opt-in and experimental (see below).

```console
$ pipx install --force "video-beep-remover[context]"   # adds PyTorch and transformers
$ vbr scan movie.mkv --context --review-srt
```

- **Harmless uses.** Each listed word gets a verdict: `profane`, `harmless` (e.g. "the nine circles of hell", or a donkey called an ass) or `unsure`. Only words listed in `context.ambiguous` are checked; the rest are profane by definition. A use is called harmless only when a small language model (the *judge*) says so *and* a toxicity classifier finds the line clean. Anything unsure counts as profane.
- **Sexual lines.** Lines that look sexual are listed, with their evidence:
  - a classifier score for explicit lines;
  - phrases from the `sexual` word category, such as "have sex" and "sleep with";
  - SDH sound descriptions such as `[moaning]`;
  - the judge, for innuendo.

  Phrases with an innocent sense too ("hook up the printer") only make a line *possibly* sexual on their own.
- **The `sexual` category** of phrases is off by default. Turn it on (`[lexicon.categories.sexual] enabled = true`) to mute its phrases like any listed word.
- **GPU or CPU.** The classifier (about 500 MB) is fast on a CPU. The judge (Qwen3-4B-Instruct by default) runs by default only on an NVIDIA GPU with about 8 GB of memory. On a CPU it takes 20–35 s per question, so without a GPU vbr skips it. Set `context.judge` to a model name to run one anyway, or to `""` to never run one. Without a judge, nothing is called harmless, and an ambiguous word such as "hell" is never replaced, only muted.
- **A judge online.** Without a big enough GPU, `context.judge = "api"` asks a service instead, set in `[context.api]`:
  - `provider = "gemini"` (the default): Google's Gemini API, `gemini-3.5-flash-lite`. Its free tier needs only a key from Google AI Studio; Google may use what is sent on the free tier to improve its products. The free tier limits requests per minute; vbr waits as long as Gemini asks.
  - `provider = "jev"`: Jev's decision API, which answers each question as choices with probabilities. A use it is less than 70 % sure of stays unsure.
  - `provider = "openai"`: any OpenAI-compatible chat API, such as OpenRouter or a local Ollama server, with its `url` and `model`.

  The key comes from `VBR_JUDGE_API_KEY` by default. A question the service can't answer leaves the word unsure, so it is muted; a key it rejects stops the run.

  ```console
  $ export VBR_JUDGE_API_KEY=...          # e.g. from https://aistudio.google.com/apikey
  $ vbr clean movie.mkv --replace --review-srt   # with judge = "api" in [context]
  ```
- **Privacy.** The models run locally, and download once. With `context.judge = "api"`, each line the judge is asked about is sent to the service with its neighbouring lines; no audio and no file names. `--offline` turns that off.
- **Acting on the verdicts (experimental).** Two settings in `[context]` change what is muted:
  - `harmless = "keep"` leaves uses judged harmless audible. It needs the judge, since nothing else calls a use harmless. Only a subtitle line that was actually heard can show a use as harmless, so a note slipped into downloaded subtitles ("the word is used harmlessly here") cannot keep a word.
  - `sexual = "mute"` mutes each whole line flagged as sexual, from its first word to its last. A subtitle line gets a short transcription of its own to find them. Lines only *possibly* sexual are not muted.

  Both are measured only on a small labelled set so far, so vbr warns when they are on. Check what they did in the review subtitles (`--review-srt`): `[kept] hell (probably harmless: place)`, `[muted] sexual line (…)`.

On a labelled set of 71 lines, with the judge, it recognized 10 of 16 harmless uses and called no profane use harmless. It flagged explicit lines, but found little innuendo. [Appendix D of the design](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md#appendix-d-context-analysis-measurements) has the numbers, and `scripts/evaluate_context.py` measures your own lines (see [Evaluate](#evaluate)).

## Voice replacement (experimental)

`--replace` says a milder word in place of a listed one, in the speaker's own voice, instead of muting it: "That was a damn fine cup of coffee" becomes "That was a darn fine cup of coffee".

```console
$ pipx install --force "video-beep-remover[voice]"
$ vbr clean movie.mkv --replace --review-srt
```

For each word:

1. **Is there a fitting substitute?** Context analysis runs with it. The use must be profane (so an ambiguous word needs the judge), its line not sexual, and the word not shouted or whispered. The substitutes come from `[replace.substitutes]` in the config, such as `hell = ["heck"]` or `"*fuck*" = ["freaking", "frick", "fricking", "fudge"]`. The judge picks one, or, without a judge, a term with a single substitute uses it.
2. **Say it again.** The dialogue is separated from music and effects ([Demucs](https://github.com/adefossez/demucs)). Then [F5-TTS](https://github.com/SWivid/F5-TTS) speaks the word's span again inside the sentence, which keeps the voice, pace and timing. The music and effects under the word stay.
3. **Check it.** Whisper must hear the substitute, and no listed word; the new word must sound like the speaker about as much as the old one did. Anything that fails is muted as usual.

The span stays muted in other audio tracks, in the EDL and in `vbr render`. The review subtitles show `[replaced] damn → darn`, and the report lists every attempt with its outcome.

It is experimental and slow on a CPU: about 40 s a word, so a GPU is effectively required for a film. To fit a small GPU, only one large model (Whisper, the context models, Demucs or F5-TTS) is in memory at a time; with plenty of GPU memory, `models.keep_loaded = true` keeps them all loaded, which saves a few seconds per file. F5-TTS's model weights are licensed for non-commercial use only: this is for personal viewing copies. Everything runs locally.

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
$ ruff check . && ruff format --check . && python -m mypy
```

[CLAUDE.md](https://github.com/AshitakaLax/video-beep-remover/blob/main/CLAUDE.md) is a short guide to the code for people and coding agents: where each part lives, the conventions, and the traps.

## Evaluate

`scripts/evaluate.py` scores vbr against clips whose listed words are annotated with their times (`<clip>.truth.json` next to each clip; the script's docstring has the format). It runs every strategy and reports recall, precision, how far detected word edges are from the annotated ones, extra muted time, audio transcribed and wall time. `--set KEY=VALUE` compares settings.

Real film clips can't be shared, so `scripts/make_synthetic_set.py` builds a stand-in set from espeak-ng speech with exact word timings:

```console
$ python scripts/make_synthetic_set.py /tmp/vbr-eval
$ python scripts/evaluate.py /tmp/vbr-eval --set transcription.device=cpu
$ python scripts/evaluate.py /tmp/vbr-eval --set censor.pad_after_ms=200 --cache /tmp/vbr-eval-cache
```

Synthetic speech is far cleaner than a film's soundtrack, so this set catches regressions and systematic effects but can't tune the defaults for real films. On it, every strategy mutes every word it detects, with no false positives; `hybrid` fully mutes 94 % of the listed words, and what it misses are words the subtitles softened into ordinary words. It also showed that Whisper places word ends up to 230 ms early, which is why `censor.pad_after_ms` is 200. Details are in [Appendix C of the design](https://github.com/AshitakaLax/video-beep-remover/blob/main/docs/DESIGN.md#appendix-c-evaluation-on-the-synthetic-set).

`scripts/evaluate_context.py` scores [context analysis](#context-analysis-preview) on labelled lines, such as the 71 in `scripts/data/context_lines.jsonl` (the script's docstring has the format). It reports how many harmless uses are recognized, whether a profane use is ever called harmless, and how many sexual lines are flagged. `--no-judge` runs the rules and the classifier alone.

## License

MIT
